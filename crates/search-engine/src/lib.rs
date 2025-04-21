//! Search functionality for the AI orchestrator
//!
//! This module provides search capabilities for finding and retrieving
//! information from various data sources.

use std::sync::Arc;
use anyhow::Result;
use tracing::{info, warn, error, debug, trace};
use std::collections::{HashMap, HashSet};
use serde::{Serialize, Deserialize};
use tokio::sync::RwLock;
use async_trait::async_trait;

use common::error::Error;
use config::ConfigManager;
use performance_optimizations::PerformanceOptimizer;

/// Search query
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchQuery {
    /// Query text
    pub query: String,
    
    /// Maximum number of results
    pub max_results: usize,
    
    /// Minimum relevance score (0.0-1.0)
    pub min_score: f32,
    
    /// Data sources to search
    pub data_sources: Vec<String>,
    
    /// Filters to apply
    pub filters: HashMap<String, String>,
    
    /// Sort order
    pub sort_by: Option<String>,
    
    /// Sort direction
    pub sort_direction: SortDirection,
}

/// Sort direction
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SortDirection {
    /// Ascending order
    Ascending,
    
    /// Descending order
    Descending,
}

/// Search result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    /// Result ID
    pub id: String,
    
    /// Result title
    pub title: String,
    
    /// Result content
    pub content: String,
    
    /// Relevance score (0.0-1.0)
    pub score: f32,
    
    /// Data source
    pub source: String,
    
    /// Metadata
    pub metadata: HashMap<String, String>,
}

/// Search results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResults {
    /// Query that produced these results
    pub query: String,
    
    /// Total number of results found
    pub total_results: usize,
    
    /// Results returned
    pub results: Vec<SearchResult>,
    
    /// Search time in milliseconds
    pub search_time_ms: u64,
}

/// Search data source trait
#[async_trait]
pub trait SearchDataSource: Send + Sync {
    /// Gets the name of the data source
    fn name(&self) -> &str;
    
    /// Searches the data source
    async fn search(&self, query: &SearchQuery) -> Result<Vec<SearchResult>>;
    
    /// Gets the total number of documents in the data source
    async fn document_count(&self) -> Result<usize>;
}

/// In-memory search data source
pub struct InMemorySearchDataSource {
    /// Name of the data source
    name: String,
    
    /// Documents
    documents: RwLock<Vec<SearchDocument>>,
}

/// Search document
#[derive(Debug, Clone, Serialize, Deserialize)]
struct SearchDocument {
    /// Document ID
    id: String,
    
    /// Document title
    title: String,
    
    /// Document content
    content: String,
    
    /// Document metadata
    metadata: HashMap<String, String>,
}

impl InMemorySearchDataSource {
    /// Creates a new in-memory search data source
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            documents: RwLock::new(Vec::new()),
        }
    }
    
    /// Adds a document to the data source
    pub async fn add_document(
        &self,
        id: &str,
        title: &str,
        content: &str,
        metadata: HashMap<String, String>,
    ) -> Result<()> {
        let document = SearchDocument {
            id: id.to_string(),
            title: title.to_string(),
            content: content.to_string(),
            metadata,
        };
        
        let mut documents = self.documents.write().await;
        documents.push(document);
        
        Ok(())
    }
    
    /// Removes a document from the data source
    pub async fn remove_document(&self, id: &str) -> Result<()> {
        let mut documents = self.documents.write().await;
        
        let index = documents
            .iter()
            .position(|doc| doc.id == id)
            .ok_or_else(|| Error::NotFound(format!("Document with ID '{}' not found", id)))?;
        
        documents.remove(index);
        
        Ok(())
    }
    
    /// Clears all documents from the data source
    pub async fn clear(&self) -> Result<()> {
        let mut documents = self.documents.write().await;
        documents.clear();
        
        Ok(())
    }
}

#[async_trait]
impl SearchDataSource for InMemorySearchDataSource {
    fn name(&self) -> &str {
        &self.name
    }
    
    async fn search(&self, query: &SearchQuery) -> Result<Vec<SearchResult>> {
        let documents = self.documents.read().await;
        let mut results = Vec::new();
        
        // Simple search implementation
        for doc in documents.iter() {
            // Check if document matches filters
            let mut matches_filters = true;
            
            for (key, value) in &query.filters {
                if let Some(doc_value) = doc.metadata.get(key) {
                    if doc_value != value {
                        matches_filters = false;
                        break;
                    }
                } else {
                    matches_filters = false;
                    break;
                }
            }
            
            if !matches_filters {
                continue;
            }
            
            // Calculate relevance score
            let score = self.calculate_relevance(&query.query, doc);
            
            // Check if score meets minimum threshold
            if score < query.min_score {
                continue;
            }
            
            // Add to results
            results.push(SearchResult {
                id: doc.id.clone(),
                title: doc.title.clone(),
                content: doc.content.clone(),
                score,
                source: self.name.clone(),
                metadata: doc.metadata.clone(),
            });
        }
        
        // Sort results
        if let Some(sort_by) = &query.sort_by {
            results.sort_by(|a, b| {
                let a_value = a.metadata.get(sort_by).map(|s| s.as_str()).unwrap_or("");
                let b_value = b.metadata.get(sort_by).map(|s| s.as_str()).unwrap_or("");
                
                match query.sort_direction {
                    SortDirection::Ascending => a_value.cmp(b_value),
                    SortDirection::Descending => b_value.cmp(a_value),
                }
            });
        } else {
            // Sort by score
            results.sort_by(|a, b| {
                match query.sort_direction {
                    SortDirection::Ascending => a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal),
                    SortDirection::Descending => b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal),
                }
            });
        }
        
        // Limit results
        if results.len() > query.max_results {
            results.truncate(query.max_results);
        }
        
        Ok(results)
    }
    
    async fn document_count(&self) -> Result<usize> {
        let documents = self.documents.read().await;
        Ok(documents.len())
    }
}

impl InMemorySearchDataSource {
    /// Calculates relevance score for a document
    fn calculate_relevance(&self, query: &str, document: &SearchDocument) -> f32 {
        // Simple TF-IDF-like scoring
        let query_terms: HashSet<String> = query
            .to_lowercase()
            .split_whitespace()
            .map(|s| s.to_string())
            .collect();
        
        let title_terms: HashSet<String> = document.title
            .to_lowercase()
            .split_whitespace()
            .map(|s| s.to_string())
            .collect();
        
        let content_terms: HashSet<String> = document.content
            .to_lowercase()
            .split_whitespace()
            .map(|s| s.to_string())
            .collect();
        
        // Calculate term matches
        let title_matches = query_terms.intersection(&title_terms).count();
        let content_matches = query_terms.intersection(&content_terms).count();
        
        // Calculate score
        let title_score = if !title_terms.is_empty() {
            (title_matches as f32) / (title_terms.len() as f32)
        } else {
            0.0
        };
        
        let content_score = if !content_terms.is_empty() {
            (content_matches as f32) / (content_terms.len() as f32)
        } else {
            0.0
        };
        
        // Weight title matches more heavily
        let score = (title_score * 0.6) + (content_score * 0.4);
        
        score
    }
}

/// Vector search data source
pub struct VectorSearchDataSource {
    /// Name of the data source
    name: String,
    
    /// Documents
    documents: RwLock<Vec<VectorSearchDocument>>,
    
    /// Embedding function
    embedding_fn: Box<dyn Fn(&str) -> Vec<f32> + Send + Sync>,
}

/// Vector search document
#[derive(Debug, Clone)]
struct VectorSearchDocument {
    /// Document ID
    id: String,
    
    /// Document title
    title: String,
    
    /// Document content
    content: String,
    
    /// Document embedding
    embedding: Vec<f32>,
    
    /// Document metadata
    metadata: HashMap<String, String>,
}

impl VectorSearchDataSource {
    /// Creates a new vector search data source
    pub fn new<F>(name: &str, embedding_fn: F) -> Self
    where
        F: Fn(&str) -> Vec<f32> + Send + Sync + 'static,
    {
        Self {
            name: name.to_string(),
            documents: RwLock::new(Vec::new()),
            embedding_fn: Box::new(embedding_fn),
        }
    }
    
    /// Adds a document to the data source
    pub async fn add_document(
        &self,
        id: &str,
        title: &str,
        content: &str,
        metadata: HashMap<String, String>,
    ) -> Result<()> {
        let embedding = (self.embedding_fn)(content);
        
        let document = VectorSearchDocument {
            id: id.to_string(),
            title: title.to_string(),
            content: content.to_string(),
            embedding,
            metadata,
        };
        
        let mut documents = self.documents.write().await;
        documents.push(document);
        
        Ok(())
    }
    
    /// Removes a document from the data source
    pub async fn remove_document(&self, id: &str) -> Result<()> {
        let mut documents = self.documents.write().await;
        
        let index = documents
            .iter()
            .position(|doc| doc.id == id)
            .ok_or_else(|| Error::NotFound(format!("Document with ID '{}' not found", id)))?;
        
        documents.remove(index);
        
        Ok(())
    }
    
    /// Clears all documents from the data source
    pub async fn clear(&self) -> Result<()> {
        let mut documents = self.documents.write().await;
        documents.clear();
        
        Ok(())
    }
    
    /// Calculates cosine similarity between two vectors
    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        let mut dot_product = 0.0;
        let mut a_norm = 0.0;
        let mut b_norm = 0.0;
        
        for i in 0..a.len().min(b.len()) {
            dot_product += a[i] * b[i];
            a_norm += a[i] * a[i];
            b_norm += b[i] * b[i];
        }
        
        if a_norm == 0.0 || b_norm == 0.0 {
            return 0.0;
        }
        
        dot_product / (a_norm.sqrt() * b_norm.sqrt())
    }
}

#[async_trait]
impl SearchDataSource for VectorSearchDataSource {
    fn name(&self) -> &str {
        &self.name
    }
    
    async fn search(&self, query: &SearchQuery) -> Result<Vec<SearchResult>> {
        let documents = self.documents.read().await;
        let mut results = Vec::new();
        
        // Generate query embedding
        let query_embedding = (self.embedding_fn)(&query.query);
        
        // Vector search implementation
        for doc in documents.iter() {
            // Check if document matches filters
            let mut matches_filters = true;
            
            for (key, value) in &query.filters {
                if let Some(doc_value) = doc.metadata.get(key) {
                    if doc_value != value {
                        matches_filters = false;
                        break;
                    }
                } else {
                    matches_filters = false;
                    break;
                }
            }
            
            if !matches_filters {
                continue;
            }
            
            // Calculate similarity score
            let score = Self::cosine_similarity(&query_embedding, &doc.embedding);
            
            // Check if score meets minimum threshold
            if score < query.min_score {
                continue;
            }
            
            // Add to results
            results.push(SearchResult {
                id: doc.id.clone(),
                title: doc.title.clone(),
                content: doc.content.clone(),
                score,
                source: self.name.clone(),
                metadata: doc.metadata.clone(),
            });
        }
        
        // Sort results
        if let Some(sort_by) = &query.sort_by {
            results.sort_by(|a, b| {
                let a_value = a.metadata.get(sort_by).map(|s| s.as_str()).unwrap_or("");
                let b_value = b.metadata.get(sort_by).map(|s| s.as_str()).unwrap_or("");
                
                match query.sort_direction {
                    SortDirection::Ascending => a_value.cmp(b_value),
                    SortDirection::Descending => b_value.cmp(a_value),
                }
            });
        } else {
            // Sort by score
            results.sort_by(|a, b| {
                match query.sort_direction {
                    SortDirection::Ascending => a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal),
                    SortDirection::Descending => b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal),
                }
            });
        }
        
        // Limit results
        if results.len() > query.max_results {
            results.truncate(query.max_results);
        }
        
        Ok(results)
    }
    
    async fn document_count(&self) -> Result<usize> {
        let documents = self.documents.read().await;
        Ok(documents.len())
    }
}

/// Search engine
pub struct SearchEngine {
    /// Configuration manager
    config_manager: Arc<ConfigManager>,
    
    /// Performance optimizer
    performance_optimizer: Arc<PerformanceOptimizer>,
    
    /// Data sources
    data_sources: RwLock<HashMap<String, Arc<dyn SearchDataSource>>>,
}

impl SearchEngine {
    /// Creates a new search engine
    pub fn new(
        config_manager: Arc<ConfigManager>,
        performance_optimizer: Arc<PerformanceOptimizer>,
    ) -> Self {
        Self {
            config_manager,
            performance_optimizer,
            data_sources: RwLock::new(HashMap::new()),
        }
    }
    
    /// Registers a data source
    pub async fn register_data_source(&self, data_source: Arc<dyn SearchDataSource>) -> Result<()> {
        let name = data_source.name().to_string();
        
        let mut data_sources = self.data_sources.write().await;
        
        if data_sources.contains_key(&name) {
            return Err(Error::AlreadyExists(format!("Data source '{}' already exists", name)).into());
        }
        
        data_sources.insert(name, data_source);
        
        Ok(())
    }
    
    /// Unregisters a data source
    pub async fn unregister_data_source(&self, name: &str) -> Result<()> {
        let mut data_sources = self.data_sources.write().await;
        
        if !data_sources.contains_key(name) {
            return Err(Error::NotFound(format!("Data source '{}' not found", name)).into());
        }
        
        data_sources.remove(name);
        
        Ok(())
    }
    
    /// Gets all data sources
    pub async fn get_data_sources(&self) -> Vec<String> {
        let data_sources = self.data_sources.read().await;
        data_sources.keys().cloned().collect()
    }
    
    /// Searches all data sources
    pub async fn search(&self, query: SearchQuery) -> Result<SearchResults> {
        let start_time = std::time::Instant::now();
        
        let data_sources = self.data_sources.read().await;
        let mut all_results = Vec::new();
        
        // If no data sources specified, search all
        let sources_to_search = if query.data_sources.is_empty() {
            data_sources.keys().cloned().collect()
        } else {
            query.data_sources.clone()
        };
        
        // Search each data source
        for source_name in &sources_to_search {
            if let Some(source) = data_sources.get(source_name) {
                match source.search(&query).await {
                    Ok(results) => {
                        all_results.extend(results);
                    },
                    Err(e) => {
                        warn!("Error searching data source '{}': {}", source_name, e);
                    }
                }
            }
        }
        
        // Sort results
        if let Some(sort_by) = &query.sort_by {
            all_results.sort_by(|a, b| {
                let a_value = a.metadata.get(sort_by).map(|s| s.as_str()).unwrap_or("");
                let b_value = b.metadata.get(sort_by).map(|s| s.as_str()).unwrap_or("");
                
                match query.sort_direction {
                    SortDirection::Ascending => a_value.cmp(b_value),
                    SortDirection::Descending => b_value.cmp(a_value),
                }
            });
        } else {
            // Sort by score
            all_results.sort_by(|a, b| {
                match query.sort_direction {
                    SortDirection::Ascending => a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal),
                    SortDirection::Descending => b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal),
                }
            });
        }
        
        // Limit results
        let total_results = all_results.len();
        
        if all_results.len() > query.max_results {
            all_results.truncate(query.max_results);
        }
        
        let search_time_ms = start_time.elapsed().as_millis() as u64;
        
        Ok(SearchResults {
            query: query.query,
            total_results,
            results: all_results,
            search_time_ms,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    // These tests would require mocking the dependencies
    // For now, we just have placeholder tests
    
    #[test]
    fn test_search_engine() {
        // Placeholder test
        assert!(true);
    }
}

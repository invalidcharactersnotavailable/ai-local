//! Model metadata management
//!
//! This module provides functionality for managing model metadata,
//! including extraction, validation, and indexing.

use std::sync::Arc;
use anyhow::Result;
use tracing::{info, warn, error, debug};
use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use std::path::{Path, PathBuf};

use common::error::Error;
use common::models::{Model, ModelType};
use config::ConfigManager;

/// Model metadata manager
pub struct MetadataManager {
    /// Configuration manager
    config_manager: Arc<ConfigManager>,
    
    /// Metadata cache
    metadata_cache: dashmap::DashMap<String, ModelMetadata>,
    
    /// Models directory
    models_dir: PathBuf,
}

/// Detailed model metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    /// Model ID
    pub id: String,
    
    /// Model name
    pub name: String,
    
    /// Model version
    pub version: String,
    
    /// Model description
    pub description: String,
    
    /// Model architecture
    pub architecture: String,
    
    /// Model family
    pub family: String,
    
    /// Model parameter count
    pub parameter_count: u64,
    
    /// Model vocabulary size
    pub vocabulary_size: usize,
    
    /// Model context window size
    pub context_window: usize,
    
    /// Model embedding size
    pub embedding_size: usize,
    
    /// Model attention heads
    pub attention_heads: usize,
    
    /// Model layers
    pub layers: usize,
    
    /// Model license
    pub license: String,
    
    /// Model tags
    pub tags: Vec<String>,
    
    /// Model capabilities
    pub capabilities: Vec<String>,
    
    /// Model languages
    pub languages: Vec<String>,
    
    /// Model creation date
    pub creation_date: String,
    
    /// Model authors
    pub authors: Vec<String>,
    
    /// Model repository URL
    pub repository_url: Option<String>,
    
    /// Model paper URL
    pub paper_url: Option<String>,
    
    /// Model homepage URL
    pub homepage_url: Option<String>,
    
    /// Additional metadata
    pub additional: HashMap<String, String>,
}

impl MetadataManager {
    /// Creates a new metadata manager
    pub fn new(config_manager: Arc<ConfigManager>) -> Result<Self> {
        // Get models directory from config
        let models_dir = config_manager
            .get_path("model_storage_path")
            .unwrap_or_else(|_| PathBuf::from("/var/lib/ai-orchestrator/models"));
        
        Ok(Self {
            config_manager,
            metadata_cache: dashmap::DashMap::new(),
            models_dir,
        })
    }
    
    /// Extracts metadata from a model
    pub async fn extract_metadata(&self, model: &Model) -> Result<ModelMetadata> {
        let model_id = &model.id;
        
        // Check if metadata is already cached
        if let Some(metadata) = self.metadata_cache.get(model_id) {
            return Ok(metadata.clone());
        }
        
        // Get model directory
        let model_dir = self.models_dir.join(model_id);
        
        // Check if detailed metadata file exists
        let metadata_path = model_dir.join("metadata.json");
        let detailed_metadata_path = model_dir.join("detailed_metadata.json");
        
        if detailed_metadata_path.exists() {
            // Load detailed metadata
            let metadata_json = tokio::fs::read_to_string(&detailed_metadata_path).await?;
            let metadata: ModelMetadata = serde_json::from_str(&metadata_json)?;
            
            // Cache metadata
            self.metadata_cache.insert(model_id.to_string(), metadata.clone());
            
            return Ok(metadata);
        }
        
        // No detailed metadata, create from model info
        let metadata = self.create_metadata_from_model(model)?;
        
        // Save detailed metadata
        let metadata_json = serde_json::to_string_pretty(&metadata)?;
        tokio::fs::write(&detailed_metadata_path, metadata_json).await?;
        
        // Cache metadata
        self.metadata_cache.insert(model_id.to_string(), metadata.clone());
        
        Ok(metadata)
    }
    
    /// Creates metadata from model information
    fn create_metadata_from_model(&self, model: &Model) -> Result<ModelMetadata> {
        // Extract information from model
        let parameter_count = self.estimate_parameter_count(model.size);
        
        // Determine architecture based on model type
        let architecture = match model.model_type {
            ModelType::LLM => "Transformer".to_string(),
            ModelType::Diffusion => "Diffusion".to_string(),
            ModelType::Speech => "Wav2Vec".to_string(),
            ModelType::Vision => "Vision Transformer".to_string(),
            ModelType::Multimodal => "Multimodal Transformer".to_string(),
            ModelType::Custom(ref custom) => custom.clone(),
        };
        
        // Determine family based on model name
        let family = if model.name.contains("llama") {
            "LLaMA".to_string()
        } else if model.name.contains("gpt") {
            "GPT".to_string()
        } else if model.name.contains("bert") {
            "BERT".to_string()
        } else if model.name.contains("t5") {
            "T5".to_string()
        } else if model.name.contains("stable-diffusion") {
            "Stable Diffusion".to_string()
        } else {
            "Unknown".to_string()
        };
        
        // Estimate model parameters
        let embedding_size = (parameter_count / 1_000_000_000 * 128).max(768) as usize;
        let attention_heads = (embedding_size / 64).max(12);
        let layers = (parameter_count / 1_000_000_000 * 8).max(12) as usize;
        let vocabulary_size = 32000;
        let context_window = model.config.context_size;
        
        // Create metadata
        let metadata = ModelMetadata {
            id: model.id.clone(),
            name: model.name.clone(),
            version: model.version.clone(),
            description: model.description.clone(),
            architecture,
            family,
            parameter_count,
            vocabulary_size,
            context_window,
            embedding_size,
            attention_heads,
            layers,
            license: model.metadata.get("license").cloned().unwrap_or_else(|| "Unknown".to_string()),
            tags: model.metadata.get("tags")
                .map(|t| t.split(',').map(|s| s.trim().to_string()).collect())
                .unwrap_or_else(Vec::new),
            capabilities: model.metadata.get("capabilities")
                .map(|c| c.split(',').map(|s| s.trim().to_string()).collect())
                .unwrap_or_else(Vec::new),
            languages: model.metadata.get("languages")
                .map(|l| l.split(',').map(|s| s.trim().to_string()).collect())
                .unwrap_or_else(|| vec!["en".to_string()]),
            creation_date: model.metadata.get("creation_date")
                .cloned()
                .unwrap_or_else(|| model.created_at.to_rfc3339()),
            authors: model.metadata.get("authors")
                .map(|a| a.split(',').map(|s| s.trim().to_string()).collect())
                .unwrap_or_else(Vec::new),
            repository_url: model.metadata.get("repository_url").cloned(),
            paper_url: model.metadata.get("paper_url").cloned(),
            homepage_url: model.metadata.get("homepage_url").cloned(),
            additional: model.metadata.clone(),
        };
        
        Ok(metadata)
    }
    
    /// Estimates parameter count from model size
    fn estimate_parameter_count(&self, size_bytes: u64) -> u64 {
        // Assuming FP32 parameters (4 bytes per parameter)
        size_bytes / 4
    }
    
    /// Updates metadata for a model
    pub async fn update_metadata(&self, model_id: &str, metadata: &ModelMetadata) -> Result<()> {
        // Get model directory
        let model_dir = self.models_dir.join(model_id);
        
        // Check if directory exists
        if !model_dir.exists() {
            return Err(Error::NotFound(format!("Model directory not found: {}", model_id)).into());
        }
        
        // Save detailed metadata
        let detailed_metadata_path = model_dir.join("detailed_metadata.json");
        let metadata_json = serde_json::to_string_pretty(metadata)?;
        tokio::fs::write(&detailed_metadata_path, metadata_json).await?;
        
        // Update cache
        self.metadata_cache.insert(model_id.to_string(), metadata.clone());
        
        Ok(())
    }
    
    /// Gets metadata for a model
    pub async fn get_metadata(&self, model_id: &str) -> Result<ModelMetadata> {
        // Check if metadata is already cached
        if let Some(metadata) = self.metadata_cache.get(model_id) {
            return Ok(metadata.clone());
        }
        
        // Get model directory
        let model_dir = self.models_dir.join(model_id);
        
        // Check if detailed metadata file exists
        let detailed_metadata_path = model_dir.join("detailed_metadata.json");
        
        if detailed_metadata_path.exists() {
            // Load detailed metadata
            let metadata_json = tokio::fs::read_to_string(&detailed_metadata_path).await?;
            let metadata: ModelMetadata = serde_json::from_str(&metadata_json)?;
            
            // Cache metadata
            self.metadata_cache.insert(model_id.to_string(), metadata.clone());
            
            return Ok(metadata);
        }
        
        Err(Error::NotFound(format!("Metadata not found for model: {}", model_id)).into())
    }
    
    /// Searches for models matching criteria
    pub async fn search_models(&self, query: &str, tags: Option<Vec<String>>, limit: usize) -> Result<Vec<String>> {
        let mut results = Vec::new();
        
        // Normalize query
        let query = query.to_lowercase();
        
        // Search through metadata cache
        for entry in self.metadata_cache.iter() {
            let metadata = entry.value();
            
            // Check if model matches query
            let matches_query = query.is_empty() || 
                metadata.name.to_lowercase().contains(&query) ||
                metadata.description.to_lowercase().contains(&query) ||
                metadata.architecture.to_lowercase().contains(&query) ||
                metadata.family.to_lowercase().contains(&query) ||
                metadata.tags.iter().any(|t| t.to_lowercase().contains(&query)) ||
                metadata.capabilities.iter().any(|c| c.to_lowercase().contains(&query)) ||
                metadata.languages.iter().any(|l| l.to_lowercase().contains(&query));
            
            // Check if model matches tags
            let matches_tags = tags.as_ref().map_or(true, |t| {
                t.iter().all(|tag| metadata.tags.contains(tag))
            });
            
            if matches_query && matches_tags {
                results.push(metadata.id.clone());
                
                if results.len() >= limit {
                    break;
                }
            }
        }
        
        Ok(results)
    }
    
    /// Indexes all models in the repository
    pub async fn index_all_models(&self) -> Result<()> {
        info!("Indexing all models in repository");
        
        // Scan models directory
        let entries = tokio::fs::read_dir(&self.models_dir).await?;
        let mut entry_count = 0;
        
        tokio::pin!(entries);
        
        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            
            if path.is_dir() {
                let model_id = path.file_name()
                    .and_then(|name| name.to_str())
                    .ok_or_else(|| Error::Internal("Invalid model directory name".to_string()))?;
                
                // Check if metadata file exists
                let metadata_path = path.join("metadata.json");
                
                if metadata_path.exists() {
                    // Load model metadata
                    let metadata_json = tokio::fs::read_to_string(&metadata_path).await?;
                    let model: Model = serde_json::from_str(&metadata_json)?;
                    
                    // Extract detailed metadata
                    match self.extract_metadata(&model).await {
                        Ok(metadata) => {
                            debug!("Indexed metadata for model {}", model_id);
                            entry_count += 1;
                        },
                        Err(e) => {
                            warn!("Failed to extract metadata for model {}: {}", model_id, e);
                        }
                    }
                }
            }
        }
        
        info!("Indexed metadata for {} models", entry_count);
        
        Ok(())
    }
    
    /// Clears the metadata cache
    pub fn clear_cache(&self) {
        self.metadata_cache.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    // These tests would require mocking the dependencies
    // For now, we just have placeholder tests
    
    #[test]
    fn test_metadata_manager() {
        // Placeholder test
        assert!(true);
    }
}

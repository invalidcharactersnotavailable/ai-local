//! Model repository implementation
//!
//! This module provides functionality for managing the repository of AI models,
//! including storage, retrieval, and lifecycle management.

use std::sync::Arc;
use tokio::sync::RwLock;
use anyhow::Result;
use tracing::{info, warn, error, debug, trace};
use dashmap::DashMap;
use std::path::{Path, PathBuf};
use std::collections::HashMap;

use common::error::Error;
use common::models::{Model, ModelStatus, ModelType, ModelConfig, Quantization, Device};
use resource_manager::ResourceMonitor;
use storage_adapter::StorageManager;
use config::ConfigManager;

/// Repository for managing AI models
pub struct ModelRepository {
    /// Available models
    models: DashMap<String, Model>,
    
    /// Loaded models (model_id -> memory address)
    loaded_models: DashMap<String, usize>,
    
    /// Model storage path
    storage_path: PathBuf,
    
    /// Storage manager
    storage_manager: Arc<StorageManager>,
    
    /// Resource monitor
    resource_monitor: Arc<ResourceMonitor>,
    
    /// Configuration manager
    config_manager: Arc<ConfigManager>,
}

impl ModelRepository {
    /// Creates a new model repository
    pub fn new(
        config_manager: Arc<ConfigManager>,
        resource_monitor: Arc<ResourceMonitor>,
    ) -> Result<Self> {
        // Get storage path from config
        let storage_path = config_manager
            .get_path("model_storage_path")
            .unwrap_or_else(|_| PathBuf::from("/var/lib/ai-orchestrator/models"));
        
        // Create storage path if it doesn't exist
        if !storage_path.exists() {
            std::fs::create_dir_all(&storage_path)?;
        }
        
        // Create storage manager
        let storage_manager = Arc::new(StorageManager::new(
            storage_path.clone(),
            config_manager.clone(),
        )?);
        
        let repo = Self {
            models: DashMap::new(),
            loaded_models: DashMap::new(),
            storage_path,
            storage_manager,
            resource_monitor,
            config_manager,
        };
        
        // Initialize repository
        repo.initialize()?;
        
        Ok(repo)
    }
    
    /// Initializes the repository by scanning the storage path
    fn initialize(&self) -> Result<()> {
        info!("Initializing model repository at {:?}", self.storage_path);
        
        // Scan storage path for model metadata files
        let entries = std::fs::read_dir(&self.storage_path)?;
        
        for entry in entries {
            let entry = entry?;
            let path = entry.path();
            
            if path.is_dir() {
                let model_id = path.file_name()
                    .and_then(|name| name.to_str())
                    .ok_or_else(|| Error::Internal("Invalid model directory name".to_string()))?;
                
                let metadata_path = path.join("metadata.json");
                
                if metadata_path.exists() {
                    // Load model metadata
                    match std::fs::read_to_string(&metadata_path) {
                        Ok(metadata_json) => {
                            match serde_json::from_str::<Model>(&metadata_json) {
                                Ok(mut model) => {
                                    // Update model status to Available
                                    model.status = ModelStatus::Available;
                                    
                                    // Add model to repository
                                    self.models.insert(model_id.to_string(), model);
                                    
                                    debug!("Loaded model metadata for {}", model_id);
                                },
                                Err(e) => {
                                    warn!("Failed to parse metadata for model {}: {}", model_id, e);
                                }
                            }
                        },
                        Err(e) => {
                            warn!("Failed to read metadata for model {}: {}", model_id, e);
                        }
                    }
                }
            }
        }
        
        info!("Model repository initialized with {} models", self.models.len());
        
        Ok(())
    }
    
    /// Gets a model by ID
    pub async fn get_model(&self, model_id: &str) -> Result<Model> {
        match self.models.get(model_id) {
            Some(model) => Ok(model.clone()),
            None => Err(Error::NotFound(format!("Model not found: {}", model_id)).into()),
        }
    }
    
    /// Lists all models
    pub async fn list_models(&self) -> Result<Vec<Arc<Model>>> {
        let models = self.models
            .iter()
            .map(|entry| Arc::new(entry.value().clone()))
            .collect();
        
        Ok(models)
    }
    
    /// Adds a new model to the repository
    pub async fn add_model(&self, model: Model) -> Result<()> {
        // Check if model already exists
        if self.models.contains_key(&model.id) {
            return Err(Error::AlreadyExists(format!("Model already exists: {}", model.id)).into());
        }
        
        // Create model directory
        let model_dir = self.storage_path.join(&model.id);
        std::fs::create_dir_all(&model_dir)?;
        
        // Save model metadata
        let metadata_path = model_dir.join("metadata.json");
        let metadata_json = serde_json::to_string_pretty(&model)?;
        std::fs::write(&metadata_path, metadata_json)?;
        
        // Add model to repository
        self.models.insert(model.id.clone(), model);
        
        Ok(())
    }
    
    /// Updates a model in the repository
    pub async fn update_model(&self, model: Model) -> Result<()> {
        // Check if model exists
        if !self.models.contains_key(&model.id) {
            return Err(Error::NotFound(format!("Model not found: {}", model.id)).into());
        }
        
        // Update model metadata
        let model_dir = self.storage_path.join(&model.id);
        let metadata_path = model_dir.join("metadata.json");
        let metadata_json = serde_json::to_string_pretty(&model)?;
        std::fs::write(&metadata_path, metadata_json)?;
        
        // Update model in repository
        self.models.insert(model.id.clone(), model);
        
        Ok(())
    }
    
    /// Removes a model from the repository
    pub async fn remove_model(&self, model_id: &str) -> Result<()> {
        // Check if model exists
        if !self.models.contains_key(model_id) {
            return Err(Error::NotFound(format!("Model not found: {}", model_id)).into());
        }
        
        // Check if model is loaded
        if self.loaded_models.contains_key(model_id) {
            return Err(Error::InvalidArgument(format!("Model is currently loaded: {}", model_id)).into());
        }
        
        // Remove model directory
        let model_dir = self.storage_path.join(model_id);
        std::fs::remove_dir_all(&model_dir)?;
        
        // Remove model from repository
        self.models.remove(model_id);
        
        Ok(())
    }
    
    /// Downloads a model from a remote source
    pub async fn download_model(&self, model_id: &str, url: &str) -> Result<Model> {
        info!("Downloading model {} from {}", model_id, url);
        
        // Check if model already exists
        if self.models.contains_key(model_id) {
            return Err(Error::AlreadyExists(format!("Model already exists: {}", model_id)).into());
        }
        
        // Create model directory
        let model_dir = self.storage_path.join(model_id);
        std::fs::create_dir_all(&model_dir)?;
        
        // Update model status to Downloading
        let now = chrono::Utc::now();
        let mut model = Model {
            id: model_id.to_string(),
            name: model_id.to_string(),
            version: "1.0.0".to_string(),
            description: format!("Downloaded from {}", url),
            size: 0,
            model_type: ModelType::LLM,
            status: ModelStatus::Downloading,
            config: ModelConfig {
                quantization: Quantization::None,
                device: Device::CPU,
                context_size: 4096,
                tensor_parallel: false,
                pipeline_parallel: false,
                tensor_parallel_shards: 1,
                pipeline_parallel_shards: 1,
                flash_attention: false,
                continuous_batching: false,
                custom_options: HashMap::new(),
            },
            metadata: HashMap::new(),
            created_at: now,
            updated_at: now,
        };
        
        // Add model to repository
        self.models.insert(model_id.to_string(), model.clone());
        
        // Download model files
        let download_path = model_dir.join("model.bin");
        self.storage_manager.download_file(url, &download_path).await?;
        
        // Update model status to Available
        model.status = ModelStatus::Available;
        model.size = std::fs::metadata(&download_path)?.len();
        
        // Save model metadata
        let metadata_path = model_dir.join("metadata.json");
        let metadata_json = serde_json::to_string_pretty(&model)?;
        std::fs::write(&metadata_path, metadata_json)?;
        
        // Update model in repository
        self.models.insert(model_id.to_string(), model.clone());
        
        info!("Model {} downloaded successfully", model_id);
        
        Ok(model)
    }
    
    /// Loads a model into memory
    pub async fn load_model(&self, model_id: &str) -> Result<Model> {
        info!("Loading model {}", model_id);
        
        // Check if model exists
        let mut model = match self.models.get(model_id) {
            Some(model) => model.clone(),
            None => return Err(Error::NotFound(format!("Model not found: {}", model_id)).into()),
        };
        
        // Check if model is already loaded
        if self.loaded_models.contains_key(model_id) {
            debug!("Model {} is already loaded", model_id);
            return Ok(model);
        }
        
        // Update model status to Loading
        model.status = ModelStatus::Loading;
        self.models.insert(model_id.to_string(), model.clone());
        
        // Check if we have enough resources to load the model
        let model_size_bytes = model.size;
        let available_memory = self.resource_monitor.get_available_memory().await?;
        
        if model_size_bytes > available_memory {
            // Not enough memory
            model.status = ModelStatus::Error("Not enough memory to load model".to_string());
            self.models.insert(model_id.to_string(), model.clone());
            
            return Err(Error::Resource(format!(
                "Not enough memory to load model {}: required {} bytes, available {} bytes",
                model_id, model_size_bytes, available_memory
            )).into());
        }
        
        // In a real implementation, we would load the model into memory here
        // For now, we just simulate it
        
        // Simulate model loading
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        
        // Update model status to Loaded
        model.status = ModelStatus::Loaded;
        self.models.insert(model_id.to_string(), model.clone());
        
        // Add to loaded models (with a fake memory address)
        let fake_address = rand::random::<usize>();
        self.loaded_models.insert(model_id.to_string(), fake_address);
        
        info!("Model {} loaded successfully", model_id);
        
        Ok(model)
    }
    
    /// Unloads a model from memory
    pub async fn unload_model(&self, model_id: &str) -> Result<()> {
        info!("Unloading model {}", model_id);
        
        // Check if model exists
        let mut model = match self.models.get(model_id) {
            Some(model) => model.clone(),
            None => return Err(Error::NotFound(format!("Model not found: {}", model_id)).into()),
        };
        
        // Check if model is loaded
        if !self.loaded_models.contains_key(model_id) {
            debug!("Model {} is not loaded", model_id);
            return Ok(());
        }
        
        // Update model status to Unloading
        model.status = ModelStatus::Unloading;
        self.models.insert(model_id.to_string(), model.clone());
        
        // In a real implementation, we would unload the model from memory here
        // For now, we just simulate it
        
        // Simulate model unloading
        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
        
        // Update model status to Available
        model.status = ModelStatus::Available;
        self.models.insert(model_id.to_string(), model.clone());
        
        // Remove from loaded models
        self.loaded_models.remove(model_id);
        
        info!("Model {} unloaded successfully", model_id);
        
        Ok(())
    }
    
    /// Checks if a model is loaded
    pub fn is_model_loaded(&self, model_id: &str) -> bool {
        self.loaded_models.contains_key(model_id)
    }
    
    /// Gets the number of loaded models
    pub fn get_loaded_models_count(&self) -> usize {
        self.loaded_models.len()
    }
    
    /// Gets the total size of all models
    pub fn get_total_models_size(&self) -> u64 {
        self.models
            .iter()
            .map(|entry| entry.value().size)
            .sum()
    }
    
    /// Gets the total size of loaded models
    pub fn get_loaded_models_size(&self) -> u64 {
        self.loaded_models
            .iter()
            .filter_map(|entry| self.models.get(entry.key()).map(|model| model.size))
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    // These tests would require mocking the dependencies
    // For now, we just have placeholder tests
    
    #[test]
    fn test_model_repository() {
        // Placeholder test
        assert!(true);
    }
}

//! Model downloader implementation
//!
//! This module provides functionality for downloading AI models from various sources,
//! with support for efficient downloading, verification, and progress tracking.

use std::sync::Arc;
use tokio::sync::{RwLock, mpsc};
use anyhow::Result;
use tracing::{info, warn, error, debug, trace};
use std::path::{Path, PathBuf};
use std::collections::HashMap;
use tokio::io::AsyncWriteExt;
use tokio::fs::File;
use reqwest::Client;
use sha2::{Sha256, Digest};
use futures::StreamExt;
use bytes::Bytes;
use tokio_util::sync::CancellationToken;

use common::error::Error;
use common::utils::format_bytes;
use storage_adapter::StorageManager;
use config::ConfigManager;

/// Progress information for a download
#[derive(Debug, Clone)]
pub struct DownloadProgress {
    /// Model ID
    pub model_id: String,
    /// Total bytes to download
    pub total_bytes: u64,
    /// Bytes downloaded so far
    pub downloaded_bytes: u64,
    /// Download speed in bytes per second
    pub speed_bytes_per_second: u64,
    /// Estimated time remaining in seconds
    pub eta_seconds: u64,
    /// Whether the download is complete
    pub is_complete: bool,
    /// Error message if download failed
    pub error: Option<String>,
}

/// Model downloader for efficiently downloading AI models
pub struct ModelDownloader {
    /// HTTP client
    client: Client,
    
    /// Storage manager
    storage_manager: Arc<StorageManager>,
    
    /// Configuration manager
    config_manager: Arc<ConfigManager>,
    
    /// Download progress
    progress: Arc<RwLock<HashMap<String, DownloadProgress>>>,
    
    /// Active downloads (model_id -> cancellation token)
    active_downloads: Arc<RwLock<HashMap<String, CancellationToken>>>,
    
    /// Download directory
    download_dir: PathBuf,
    
    /// Chunk size for downloads
    chunk_size: usize,
    
    /// Number of concurrent downloads
    max_concurrent_downloads: usize,
    
    /// Whether to verify checksums
    verify_checksums: bool,
}

impl ModelDownloader {
    /// Creates a new model downloader
    pub fn new(
        storage_manager: Arc<StorageManager>,
        config_manager: Arc<ConfigManager>,
    ) -> Result<Self> {
        // Create HTTP client with appropriate configuration
        let client = Client::builder()
            .user_agent("AI-Orchestrator/0.1.0")
            .timeout(std::time::Duration::from_secs(300))
            .connect_timeout(std::time::Duration::from_secs(30))
            .pool_max_idle_per_host(10)
            .build()?;
        
        // Get download directory from config
        let download_dir = config_manager
            .get_path("model_download_dir")
            .unwrap_or_else(|_| PathBuf::from("/var/lib/ai-orchestrator/downloads"));
        
        // Create download directory if it doesn't exist
        if !download_dir.exists() {
            std::fs::create_dir_all(&download_dir)?;
        }
        
        // Get chunk size from config
        let chunk_size = config_manager
            .get_usize("download_chunk_size")
            .unwrap_or(1024 * 1024); // 1 MB default
        
        // Get max concurrent downloads from config
        let max_concurrent_downloads = config_manager
            .get_usize("max_concurrent_downloads")
            .unwrap_or(3);
        
        // Get verify checksums from config
        let verify_checksums = config_manager
            .get_bool("verify_model_checksums")
            .unwrap_or(true);
        
        Ok(Self {
            client,
            storage_manager,
            config_manager,
            progress: Arc::new(RwLock::new(HashMap::new())),
            active_downloads: Arc::new(RwLock::new(HashMap::new())),
            download_dir,
            chunk_size,
            max_concurrent_downloads,
            verify_checksums,
        })
    }
    
    /// Downloads a model from a URL
    pub async fn download_model(
        &self,
        model_id: &str,
        url: &str,
        expected_checksum: Option<&str>,
    ) -> Result<PathBuf> {
        info!("Starting download of model {} from {}", model_id, url);
        
        // Check if we're already downloading this model
        {
            let active_downloads = self.active_downloads.read().await;
            if active_downloads.contains_key(model_id) {
                return Err(Error::AlreadyExists(format!("Model {} is already being downloaded", model_id)).into());
            }
        }
        
        // Check if we have too many concurrent downloads
        {
            let active_downloads = self.active_downloads.read().await;
            if active_downloads.len() >= self.max_concurrent_downloads {
                return Err(Error::Resource(format!(
                    "Too many concurrent downloads (max: {})",
                    self.max_concurrent_downloads
                )).into());
            }
        }
        
        // Create cancellation token
        let cancellation_token = CancellationToken::new();
        
        // Add to active downloads
        {
            let mut active_downloads = self.active_downloads.write().await;
            active_downloads.insert(model_id.to_string(), cancellation_token.clone());
        }
        
        // Initialize progress
        {
            let mut progress = self.progress.write().await;
            progress.insert(model_id.to_string(), DownloadProgress {
                model_id: model_id.to_string(),
                total_bytes: 0,
                downloaded_bytes: 0,
                speed_bytes_per_second: 0,
                eta_seconds: 0,
                is_complete: false,
                error: None,
            });
        }
        
        // Create download path
        let download_path = self.download_dir.join(format!("{}.download", model_id));
        let final_path = self.download_dir.join(format!("{}.bin", model_id));
        
        // Start download
        let result = self.download_file(
            model_id,
            url,
            &download_path,
            expected_checksum,
            cancellation_token.clone(),
        ).await;
        
        // Remove from active downloads
        {
            let mut active_downloads = self.active_downloads.write().await;
            active_downloads.remove(model_id);
        }
        
        match result {
            Ok(_) => {
                // Rename download file to final file
                tokio::fs::rename(&download_path, &final_path).await?;
                
                // Update progress
                {
                    let mut progress = self.progress.write().await;
                    if let Some(p) = progress.get_mut(model_id) {
                        p.is_complete = true;
                    }
                }
                
                info!("Download of model {} completed successfully", model_id);
                
                Ok(final_path)
            },
            Err(e) => {
                // Update progress with error
                {
                    let mut progress = self.progress.write().await;
                    if let Some(p) = progress.get_mut(model_id) {
                        p.error = Some(e.to_string());
                    }
                }
                
                error!("Download of model {} failed: {}", model_id, e);
                
                // Clean up download file
                let _ = tokio::fs::remove_file(&download_path).await;
                
                Err(e)
            }
        }
    }
    
    /// Downloads a file from a URL with progress tracking
    async fn download_file(
        &self,
        model_id: &str,
        url: &str,
        path: &Path,
        expected_checksum: Option<&str>,
        cancellation_token: CancellationToken,
    ) -> Result<()> {
        // Create parent directory if it doesn't exist
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        
        // Send HEAD request to get content length
        let resp = self.client.head(url).send().await?;
        
        if !resp.status().is_success() {
            return Err(Error::ExternalService(format!(
                "Failed to get content length for {}: HTTP {}",
                url, resp.status()
            )).into());
        }
        
        let total_size = resp.headers()
            .get(reqwest::header::CONTENT_LENGTH)
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(0);
        
        // Update progress with total size
        {
            let mut progress = self.progress.write().await;
            if let Some(p) = progress.get_mut(model_id) {
                p.total_bytes = total_size;
            }
        }
        
        // Send GET request
        let resp = self.client.get(url).send().await?;
        
        if !resp.status().is_success() {
            return Err(Error::ExternalService(format!(
                "Failed to download {}: HTTP {}",
                url, resp.status()
            )).into());
        }
        
        // Create file
        let mut file = File::create(path).await?;
        
        // Create hasher if we need to verify checksum
        let mut hasher = if self.verify_checksums && expected_checksum.is_some() {
            Some(Sha256::new())
        } else {
            None
        };
        
        // Download file in chunks
        let mut stream = resp.bytes_stream();
        let mut downloaded = 0u64;
        let mut last_update = std::time::Instant::now();
        let mut last_downloaded = 0u64;
        
        while let Some(chunk_result) = stream.next().await {
            // Check if download was cancelled
            if cancellation_token.is_cancelled() {
                return Err(Error::Cancelled("Download cancelled".to_string()).into());
            }
            
            // Get chunk
            let chunk = chunk_result?;
            
            // Write chunk to file
            file.write_all(&chunk).await?;
            
            // Update hasher if needed
            if let Some(hasher) = hasher.as_mut() {
                hasher.update(&chunk);
            }
            
            // Update downloaded size
            downloaded += chunk.len() as u64;
            
            // Update progress every 100ms
            let now = std::time::Instant::now();
            if now.duration_since(last_update).as_millis() > 100 {
                let elapsed = now.duration_since(last_update).as_secs_f64();
                let speed = ((downloaded - last_downloaded) as f64 / elapsed) as u64;
                
                let eta = if speed > 0 && total_size > 0 {
                    (total_size - downloaded) / speed
                } else {
                    0
                };
                
                // Update progress
                {
                    let mut progress = self.progress.write().await;
                    if let Some(p) = progress.get_mut(model_id) {
                        p.downloaded_bytes = downloaded;
                        p.speed_bytes_per_second = speed;
                        p.eta_seconds = eta;
                    }
                }
                
                last_update = now;
                last_downloaded = downloaded;
                
                debug!(
                    "Downloading model {}: {} / {} ({}/s, ETA: {}s)",
                    model_id,
                    format_bytes(downloaded),
                    format_bytes(total_size),
                    format_bytes(speed),
                    eta
                );
            }
        }
        
        // Flush and close file
        file.flush().await?;
        drop(file);
        
        // Verify checksum if needed
        if let (Some(hasher), Some(expected)) = (hasher, expected_checksum) {
            let hash = hasher.finalize();
            let hash_hex = format!("{:x}", hash);
            
            if hash_hex != expected {
                return Err(Error::InvalidArgument(format!(
                    "Checksum verification failed: expected {}, got {}",
                    expected, hash_hex
                )).into());
            }
            
            debug!("Checksum verification passed for model {}", model_id);
        }
        
        Ok(())
    }
    
    /// Cancels a download
    pub async fn cancel_download(&self, model_id: &str) -> Result<()> {
        let mut active_downloads = self.active_downloads.write().await;
        
        if let Some(token) = active_downloads.get(model_id) {
            // Cancel the download
            token.cancel();
            
            // Remove from active downloads
            active_downloads.remove(model_id);
            
            info!("Download of model {} cancelled", model_id);
            
            Ok(())
        } else {
            Err(Error::NotFound(format!("No active download for model {}", model_id)).into())
        }
    }
    
    /// Gets the progress of a download
    pub async fn get_download_progress(&self, model_id: &str) -> Result<DownloadProgress> {
        let progress = self.progress.read().await;
        
        if let Some(p) = progress.get(model_id) {
            Ok(p.clone())
        } else {
            Err(Error::NotFound(format!("No download progress for model {}", model_id)).into())
        }
    }
    
    /// Gets the progress of all downloads
    pub async fn get_all_download_progress(&self) -> Result<Vec<DownloadProgress>> {
        let progress = self.progress.read().await;
        let progress_list = progress.values().cloned().collect();
        Ok(progress_list)
    }
    
    /// Clears completed downloads from the progress list
    pub async fn clear_completed_downloads(&self) -> Result<()> {
        let mut progress = self.progress.write().await;
        
        progress.retain(|_, p| !p.is_complete && p.error.is_none());
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    // These tests would require mocking the HTTP client
    // For now, we just have placeholder tests
    
    #[test]
    fn test_model_downloader() {
        // Placeholder test
        assert!(true);
    }
}

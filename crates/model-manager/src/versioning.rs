//! Model versioning management
//!
//! This module provides functionality for managing model versions,
//! including version tracking, compatibility checking, and upgrades.

use std::sync::Arc;
use anyhow::Result;
use tracing::{info, warn, error, debug};
use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use std::path::{Path, PathBuf};
use semver::{Version, VersionReq};

use common::error::Error;
use common::models::Model;
use config::ConfigManager;

/// Model version information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelVersion {
    /// Model ID
    pub model_id: String,
    
    /// Version string (semver)
    pub version: String,
    
    /// Release notes
    pub release_notes: String,
    
    /// Release date
    pub release_date: String,
    
    /// Compatible with previous versions
    pub backward_compatible: bool,
    
    /// Requires specific hardware
    pub hardware_requirements: Option<String>,
    
    /// Requires specific software
    pub software_requirements: Option<String>,
    
    /// Download URL
    pub download_url: String,
    
    /// Checksum
    pub checksum: String,
    
    /// Size in bytes
    pub size: u64,
}

/// Model versioning manager
pub struct VersioningManager {
    /// Configuration manager
    config_manager: Arc<ConfigManager>,
    
    /// Version cache (model_id -> versions)
    version_cache: dashmap::DashMap<String, Vec<ModelVersion>>,
    
    /// Models directory
    models_dir: PathBuf,
}

impl VersioningManager {
    /// Creates a new versioning manager
    pub fn new(config_manager: Arc<ConfigManager>) -> Result<Self> {
        // Get models directory from config
        let models_dir = config_manager
            .get_path("model_storage_path")
            .unwrap_or_else(|_| PathBuf::from("/var/lib/ai-orchestrator/models"));
        
        Ok(Self {
            config_manager,
            version_cache: dashmap::DashMap::new(),
            models_dir,
        })
    }
    
    /// Gets all versions for a model
    pub async fn get_versions(&self, model_id: &str) -> Result<Vec<ModelVersion>> {
        // Check if versions are already cached
        if let Some(versions) = self.version_cache.get(model_id) {
            return Ok(versions.clone());
        }
        
        // Get model directory
        let model_dir = self.models_dir.join(model_id);
        
        // Check if versions file exists
        let versions_path = model_dir.join("versions.json");
        
        if versions_path.exists() {
            // Load versions
            let versions_json = tokio::fs::read_to_string(&versions_path).await?;
            let versions: Vec<ModelVersion> = serde_json::from_str(&versions_json)?;
            
            // Cache versions
            self.version_cache.insert(model_id.to_string(), versions.clone());
            
            return Ok(versions);
        }
        
        // No versions file, create one with current version
        let model_path = model_dir.join("metadata.json");
        
        if model_path.exists() {
            // Load model metadata
            let model_json = tokio::fs::read_to_string(&model_path).await?;
            let model: Model = serde_json::from_str(&model_json)?;
            
            // Create version
            let version = ModelVersion {
                model_id: model_id.to_string(),
                version: model.version.clone(),
                release_notes: "Initial version".to_string(),
                release_date: model.created_at.to_rfc3339(),
                backward_compatible: true,
                hardware_requirements: None,
                software_requirements: None,
                download_url: model.metadata.get("download_url").cloned().unwrap_or_default(),
                checksum: model.metadata.get("checksum").cloned().unwrap_or_default(),
                size: model.size,
            };
            
            // Save versions
            let versions = vec![version.clone()];
            let versions_json = serde_json::to_string_pretty(&versions)?;
            tokio::fs::write(&versions_path, versions_json).await?;
            
            // Cache versions
            self.version_cache.insert(model_id.to_string(), versions.clone());
            
            return Ok(versions);
        }
        
        Err(Error::NotFound(format!("Model not found: {}", model_id)).into())
    }
    
    /// Gets a specific version of a model
    pub async fn get_version(&self, model_id: &str, version: &str) -> Result<ModelVersion> {
        // Get all versions
        let versions = self.get_versions(model_id).await?;
        
        // Find the requested version
        for v in versions {
            if v.version == version {
                return Ok(v);
            }
        }
        
        Err(Error::NotFound(format!("Version {} not found for model {}", version, model_id)).into())
    }
    
    /// Gets the latest version of a model
    pub async fn get_latest_version(&self, model_id: &str) -> Result<ModelVersion> {
        // Get all versions
        let versions = self.get_versions(model_id).await?;
        
        if versions.is_empty() {
            return Err(Error::NotFound(format!("No versions found for model {}", model_id)).into());
        }
        
        // Parse versions and find the latest
        let mut latest_version = None;
        let mut latest_semver = None;
        
        for version in &versions {
            match Version::parse(&version.version) {
                Ok(semver) => {
                    if latest_semver.is_none() || semver > latest_semver.unwrap() {
                        latest_semver = Some(semver);
                        latest_version = Some(version.clone());
                    }
                },
                Err(e) => {
                    warn!("Failed to parse version {}: {}", version.version, e);
                }
            }
        }
        
        match latest_version {
            Some(version) => Ok(version),
            None => Err(Error::Internal(format!("No valid versions found for model {}", model_id)).into()),
        }
    }
    
    /// Adds a new version for a model
    pub async fn add_version(&self, version: ModelVersion) -> Result<()> {
        let model_id = &version.model_id;
        
        // Get model directory
        let model_dir = self.models_dir.join(model_id);
        
        // Check if directory exists
        if !model_dir.exists() {
            return Err(Error::NotFound(format!("Model directory not found: {}", model_id)).into());
        }
        
        // Get existing versions
        let mut versions = match self.get_versions(model_id).await {
            Ok(v) => v,
            Err(_) => Vec::new(),
        };
        
        // Check if version already exists
        for v in &versions {
            if v.version == version.version {
                return Err(Error::AlreadyExists(format!(
                    "Version {} already exists for model {}",
                    version.version, model_id
                )).into());
            }
        }
        
        // Add new version
        versions.push(version);
        
        // Sort versions by semver
        versions.sort_by(|a, b| {
            let a_semver = Version::parse(&a.version).unwrap_or_else(|_| Version::new(0, 0, 0));
            let b_semver = Version::parse(&b.version).unwrap_or_else(|_| Version::new(0, 0, 0));
            b_semver.cmp(&a_semver) // Descending order
        });
        
        // Save versions
        let versions_path = model_dir.join("versions.json");
        let versions_json = serde_json::to_string_pretty(&versions)?;
        tokio::fs::write(&versions_path, versions_json).await?;
        
        // Update cache
        self.version_cache.insert(model_id.to_string(), versions);
        
        Ok(())
    }
    
    /// Checks if a model version is compatible with a requirement
    pub fn is_compatible(&self, model_id: &str, version: &str, requirement: &str) -> Result<bool> {
        // Parse version
        let version = match Version::parse(version) {
            Ok(v) => v,
            Err(e) => return Err(Error::InvalidArgument(format!(
                "Invalid version format for {}: {}", version, e
            )).into()),
        };
        
        // Parse requirement
        let req = match VersionReq::parse(requirement) {
            Ok(r) => r,
            Err(e) => return Err(Error::InvalidArgument(format!(
                "Invalid version requirement format for {}: {}", requirement, e
            )).into()),
        };
        
        Ok(req.matches(&version))
    }
    
    /// Checks if an upgrade is available for a model
    pub async fn check_for_upgrade(&self, model_id: &str, current_version: &str) -> Result<Option<ModelVersion>> {
        // Get latest version
        let latest = self.get_latest_version(model_id).await?;
        
        // Parse versions
        let current = match Version::parse(current_version) {
            Ok(v) => v,
            Err(e) => return Err(Error::InvalidArgument(format!(
                "Invalid version format for {}: {}", current_version, e
            )).into()),
        };
        
        let latest_semver = match Version::parse(&latest.version) {
            Ok(v) => v,
            Err(e) => return Err(Error::Internal(format!(
                "Invalid version format for latest version {}: {}", latest.version, e
            )).into()),
        };
        
        // Check if upgrade is available
        if latest_semver > current {
            Ok(Some(latest))
        } else {
            Ok(None)
        }
    }
    
    /// Gets upgrade path from one version to another
    pub async fn get_upgrade_path(&self, model_id: &str, from_version: &str, to_version: &str) -> Result<Vec<ModelVersion>> {
        // Get all versions
        let all_versions = self.get_versions(model_id).await?;
        
        // Parse versions
        let from = match Version::parse(from_version) {
            Ok(v) => v,
            Err(e) => return Err(Error::InvalidArgument(format!(
                "Invalid version format for {}: {}", from_version, e
            )).into()),
        };
        
        let to = match Version::parse(to_version) {
            Ok(v) => v,
            Err(e) => return Err(Error::InvalidArgument(format!(
                "Invalid version format for {}: {}", to_version, e
            )).into()),
        };
        
        // Check if upgrade is needed
        if from >= to {
            return Ok(Vec::new());
        }
        
        // Find all versions between from and to
        let mut upgrade_path = Vec::new();
        
        for version in &all_versions {
            match Version::parse(&version.version) {
                Ok(semver) => {
                    if semver > from && semver <= to {
                        upgrade_path.push(version.clone());
                    }
                },
                Err(_) => continue,
            }
        }
        
        // Sort upgrade path
        upgrade_path.sort_by(|a, b| {
            let a_semver = Version::parse(&a.version).unwrap_or_else(|_| Version::new(0, 0, 0));
            let b_semver = Version::parse(&b.version).unwrap_or_else(|_| Version::new(0, 0, 0));
            a_semver.cmp(&b_semver) // Ascending order
        });
        
        Ok(upgrade_path)
    }
    
    /// Clears the version cache
    pub fn clear_cache(&self) {
        self.version_cache.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    // These tests would require mocking the dependencies
    // For now, we just have placeholder tests
    
    #[test]
    fn test_versioning_manager() {
        // Placeholder test
        assert!(true);
    }
}

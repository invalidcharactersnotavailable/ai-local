//! Persistent storage management for AI Orchestrator
//!
//! This crate provides functionality for managing persistent storage
//! of models, configurations, and other data.

pub mod manager;
pub mod filesystem;
pub mod compression;
pub mod cache;
pub mod backup;
pub mod integrity;

// Re-export commonly used types
pub use manager::StorageManager;
pub use filesystem::FilesystemAdapter;
pub use compression::CompressionManager;
pub use cache::CacheManager;

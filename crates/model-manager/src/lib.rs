//! Model lifecycle management for AI Orchestrator
//!
//! This crate provides functionality for managing the lifecycle of AI models,
//! including downloading, loading, unloading, and deletion.

pub mod repository;
pub mod downloader;
pub mod loader;
pub mod config;
pub mod metadata;
pub mod versioning;

// Re-export commonly used types
pub use repository::ModelRepository;
pub use downloader::ModelDownloader;
pub use loader::ModelLoader;
pub use config::ModelConfigManager;

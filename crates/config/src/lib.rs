//! Configuration management for AI Orchestrator
//!
//! This crate provides functionality for managing configuration settings
//! for the AI Orchestrator, with support for different configuration sources.

pub mod manager;
pub mod sources;
pub mod validation;
pub mod defaults;
pub mod schema;
pub mod overrides;
pub mod environment;

// Re-export commonly used types
pub use manager::ConfigManager;
pub use sources::{ConfigSource, FileSource, EnvSource};
pub use validation::ConfigValidator;
pub use defaults::DefaultConfig;

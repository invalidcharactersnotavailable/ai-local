//! Common utilities and types for AI Orchestrator
//! 
//! This crate provides shared functionality used across the AI Orchestrator system,
//! including error types, common data structures, and utility functions.

pub mod error;
pub mod models;
pub mod types;
pub mod utils;

// Re-export commonly used types
pub use error::{Error, Result};
pub use models::*;
pub use types::*;

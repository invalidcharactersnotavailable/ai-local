//! Model inference execution for AI Orchestrator
//!
//! This crate provides functionality for executing AI model inference
//! with optimized performance across different hardware configurations.

pub mod engine;
pub mod optimization;
pub mod batching;
pub mod precision;
pub mod memory;
pub mod kernels;
pub mod pipeline;

// Re-export commonly used types
pub use engine::InferenceEngine;
pub use batching::BatchManager;
pub use precision::PrecisionManager;
pub use memory::MemoryManager;

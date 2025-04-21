//! Core orchestration logic for AI Orchestrator
//!
//! This crate provides the central coordination system for the AI Orchestrator,
//! managing all other components and orchestrating their interactions.

pub mod engine;
pub mod coordinator;
pub mod lifecycle;
pub mod state;
pub mod metrics;

// Re-export commonly used types
pub use engine::OrchestratorEngine;
pub use coordinator::Coordinator;
pub use lifecycle::LifecycleManager;
pub use state::OrchestratorState;

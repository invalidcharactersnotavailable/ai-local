//! Dynamic scaling and adaptation for AI Orchestrator
//!
//! This crate provides functionality for dynamic scaling and adaptation
//! based on workload and available resources.

pub mod scaler;
pub mod adapter;
pub mod balancer;
pub mod fallback;
pub mod predictor;
pub mod mode_switcher;

// Re-export commonly used types
pub use scaler::DynamicScaler;
pub use adapter::ResourceAdapter;
pub use balancer::LoadBalancer;
pub use fallback::FallbackManager;
pub use predictor::PerformancePredictor;
pub use mode_switcher::ModeSwitcher;

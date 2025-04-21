//! Hardware detection and benchmarking for AI Orchestrator
//!
//! This crate provides functionality for detecting hardware capabilities
//! and benchmarking system performance for optimal AI model selection.

pub mod detector;
pub mod profiler;
pub mod benchmark;
pub mod capabilities;
pub mod recommendations;
pub mod compatibility;

// Re-export commonly used types
pub use detector::HardwareDetector;
pub use profiler::SystemProfiler;
pub use benchmark::BenchmarkRunner;
pub use capabilities::HardwareCapabilities;

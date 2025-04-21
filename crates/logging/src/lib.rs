//! Logging and monitoring for AI Orchestrator
//!
//! This crate provides functionality for logging and monitoring the AI Orchestrator,
//! with structured logging and configurable output destinations.

pub mod logger;
pub mod formatter;
pub mod appender;
pub mod filter;
pub mod metrics;
pub mod tracing;
pub mod events;

// Re-export commonly used types
pub use logger::Logger;
pub use formatter::LogFormatter;
pub use appender::LogAppender;
pub use filter::LogFilter;
pub use events::EventLogger;

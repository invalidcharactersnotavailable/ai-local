//! Common types for AI Orchestrator
//!
//! This module defines common types used throughout the AI Orchestrator system.

use std::fmt;
use std::str::FromStr;
use serde::{Deserialize, Serialize};

/// Operational mode for the AI Orchestrator
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OperationalMode {
    /// Standard mode - balanced performance and resource usage
    Standard,
    /// High-effort mode - maximum performance, higher resource usage
    HighEffort,
}

impl fmt::Display for OperationalMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OperationalMode::Standard => write!(f, "Standard"),
            OperationalMode::HighEffort => write!(f, "HighEffort"),
        }
    }
}

impl FromStr for OperationalMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "standard" => Ok(OperationalMode::Standard),
            "higheffort" | "high-effort" | "high_effort" => Ok(OperationalMode::HighEffort),
            _ => Err(format!("Unknown operational mode: {}", s)),
        }
    }
}

/// Resource requirements for a model or task
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRequirements {
    /// Minimum CPU cores required
    pub min_cpu_cores: usize,
    /// Minimum memory in bytes required
    pub min_memory_bytes: u64,
    /// Minimum GPU memory in bytes required (if GPU is used)
    pub min_gpu_memory_bytes: Option<u64>,
    /// Minimum disk space in bytes required
    pub min_disk_bytes: u64,
    /// Whether GPU is required
    pub requires_gpu: bool,
    /// Required CPU features (e.g., AVX, AVX2, AVX-512)
    pub required_cpu_features: Vec<String>,
    /// Required GPU features (e.g., CUDA compute capability)
    pub required_gpu_features: Vec<String>,
}

/// Performance metrics for the AI Orchestrator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// CPU usage percentage
    pub cpu_usage_percent: f32,
    /// Memory usage in bytes
    pub memory_usage_bytes: u64,
    /// GPU usage percentage (if GPU is used)
    pub gpu_usage_percent: Option<f32>,
    /// GPU memory usage in bytes (if GPU is used)
    pub gpu_memory_usage_bytes: Option<u64>,
    /// Disk usage in bytes
    pub disk_usage_bytes: u64,
    /// Network usage in bytes per second
    pub network_usage_bytes_per_second: u64,
    /// Inference latency in milliseconds
    pub inference_latency_ms: f64,
    /// Inference throughput in requests per second
    pub inference_throughput_rps: f64,
    /// Queue length
    pub queue_length: usize,
    /// Active tasks count
    pub active_tasks_count: usize,
}

/// Version information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Version {
    /// Major version
    pub major: u32,
    /// Minor version
    pub minor: u32,
    /// Patch version
    pub patch: u32,
    /// Pre-release version (e.g., "alpha.1", "beta.2")
    pub pre_release: Option<String>,
    /// Build metadata
    pub build_metadata: Option<String>,
}

impl fmt::Display for Version {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)?;
        if let Some(pre) = &self.pre_release {
            write!(f, "-{}", pre)?;
        }
        if let Some(build) = &self.build_metadata {
            write!(f, "+{}", build)?;
        }
        Ok(())
    }
}

/// Log level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LogLevel {
    /// Trace level
    Trace,
    /// Debug level
    Debug,
    /// Info level
    Info,
    /// Warn level
    Warn,
    /// Error level
    Error,
}

impl fmt::Display for LogLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LogLevel::Trace => write!(f, "TRACE"),
            LogLevel::Debug => write!(f, "DEBUG"),
            LogLevel::Info => write!(f, "INFO"),
            LogLevel::Warn => write!(f, "WARN"),
            LogLevel::Error => write!(f, "ERROR"),
        }
    }
}

impl FromStr for LogLevel {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_uppercase().as_str() {
            "TRACE" => Ok(LogLevel::Trace),
            "DEBUG" => Ok(LogLevel::Debug),
            "INFO" => Ok(LogLevel::Info),
            "WARN" => Ok(LogLevel::Warn),
            "ERROR" => Ok(LogLevel::Error),
            _ => Err(format!("Unknown log level: {}", s)),
        }
    }
}

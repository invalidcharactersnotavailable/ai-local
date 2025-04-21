//! Common data models for AI Orchestrator
//!
//! This module defines the common data models used throughout the AI Orchestrator system.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;
use chrono::{DateTime, Utc};

/// Model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Model {
    /// Unique identifier for the model
    pub id: String,
    /// Model name
    pub name: String,
    /// Model version
    pub version: String,
    /// Model description
    pub description: String,
    /// Model size in parameters
    pub size: u64,
    /// Model type (e.g., LLM, diffusion, etc.)
    pub model_type: ModelType,
    /// Model status
    pub status: ModelStatus,
    /// Model configuration
    pub config: ModelConfig,
    /// Model metadata
    pub metadata: HashMap<String, String>,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Last updated timestamp
    pub updated_at: DateTime<Utc>,
}

/// Model type
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ModelType {
    /// Large Language Model
    LLM,
    /// Diffusion Model
    Diffusion,
    /// Speech Model
    Speech,
    /// Vision Model
    Vision,
    /// Multimodal Model
    Multimodal,
    /// Custom Model
    Custom(String),
}

/// Model status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ModelStatus {
    /// Model is available for use
    Available,
    /// Model is being downloaded
    Downloading,
    /// Model is being loaded
    Loading,
    /// Model is loaded and ready for inference
    Loaded,
    /// Model is being unloaded
    Unloading,
    /// Model is being deleted
    Deleting,
    /// Model is in error state
    Error(String),
}

/// Model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Quantization type
    pub quantization: Quantization,
    /// Device to run the model on
    pub device: Device,
    /// Context window size
    pub context_size: usize,
    /// Whether to use tensor parallelism
    pub tensor_parallel: bool,
    /// Whether to use pipeline parallelism
    pub pipeline_parallel: bool,
    /// Number of tensor parallel shards
    pub tensor_parallel_shards: usize,
    /// Number of pipeline parallel shards
    pub pipeline_parallel_shards: usize,
    /// Whether to use flash attention
    pub flash_attention: bool,
    /// Whether to use continuous batching
    pub continuous_batching: bool,
    /// Custom configuration options
    pub custom_options: HashMap<String, String>,
}

/// Quantization type
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum Quantization {
    /// No quantization (FP32)
    None,
    /// FP16 precision
    FP16,
    /// BF16 precision
    BF16,
    /// INT8 quantization
    INT8,
    /// INT4 quantization
    INT4,
    /// Custom quantization
    Custom(String),
}

/// Device to run the model on
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum Device {
    /// CPU
    CPU,
    /// CUDA GPU
    CUDA(usize),
    /// ROCm GPU
    ROCm(usize),
    /// Multiple GPUs
    MultiGPU(Vec<usize>),
    /// Custom device
    Custom(String),
}

/// Task information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Task {
    /// Unique identifier for the task
    pub id: Uuid,
    /// Task type
    pub task_type: TaskType,
    /// Task status
    pub status: TaskStatus,
    /// Task priority
    pub priority: TaskPriority,
    /// Model ID
    pub model_id: String,
    /// Input data
    pub input: TaskInput,
    /// Output data
    pub output: Option<TaskOutput>,
    /// Task configuration
    pub config: TaskConfig,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Start timestamp
    pub started_at: Option<DateTime<Utc>>,
    /// Completion timestamp
    pub completed_at: Option<DateTime<Utc>>,
    /// Error message if task failed
    pub error: Option<String>,
    /// Task metadata
    pub metadata: HashMap<String, String>,
}

/// Task type
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum TaskType {
    /// Inference task
    Inference,
    /// Fine-tuning task
    FineTuning,
    /// Evaluation task
    Evaluation,
    /// Export task
    Export,
    /// Custom task
    Custom(String),
}

/// Task status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum TaskStatus {
    /// Task is created but not yet queued
    Created,
    /// Task is queued for execution
    Queued,
    /// Task is running
    Running,
    /// Task is completed successfully
    Completed,
    /// Task failed
    Failed,
    /// Task is cancelled
    Cancelled,
    /// Task is paused
    Paused,
}

/// Task priority
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum TaskPriority {
    /// Low priority
    Low,
    /// Normal priority
    Normal,
    /// High priority
    High,
    /// Critical priority
    Critical,
}

/// Task input
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TaskInput {
    /// Text input
    Text(String),
    /// File input
    File(String),
    /// API input
    API {
        /// API endpoint
        endpoint: String,
        /// API key
        key: Option<String>,
    },
    /// Custom input
    Custom(HashMap<String, String>),
}

/// Task output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TaskOutput {
    /// Text output
    Text(String),
    /// File output
    File(String),
    /// JSON output
    JSON(serde_json::Value),
    /// Custom output
    Custom(HashMap<String, String>),
}

/// Task configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskConfig {
    /// Execution mode
    pub execution_mode: ExecutionMode,
    /// Maximum execution time in seconds
    pub max_execution_time: Option<u64>,
    /// Whether to notify on completion
    pub notify_on_completion: bool,
    /// Custom configuration options
    pub custom_options: HashMap<String, String>,
}

/// Execution mode
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ExecutionMode {
    /// Standard mode
    Standard,
    /// High-effort mode
    HighEffort,
    /// Custom mode
    Custom(String),
}

/// System resource information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemResources {
    /// CPU information
    pub cpu: CpuInfo,
    /// Memory information
    pub memory: MemoryInfo,
    /// GPU information
    pub gpus: Vec<GpuInfo>,
    /// Disk information
    pub disk: DiskInfo,
    /// Network information
    pub network: NetworkInfo,
    /// System performance score
    pub performance_score: u32,
}

/// CPU information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuInfo {
    /// CPU model
    pub model: String,
    /// Number of physical cores
    pub physical_cores: usize,
    /// Number of logical cores
    pub logical_cores: usize,
    /// CPU architecture
    pub architecture: String,
    /// CPU frequency in MHz
    pub frequency_mhz: u32,
    /// CPU cache size in KB
    pub cache_size_kb: u32,
    /// CPU features
    pub features: Vec<String>,
    /// CPU usage percentage
    pub usage_percent: f32,
}

/// Memory information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryInfo {
    /// Total memory in bytes
    pub total_bytes: u64,
    /// Available memory in bytes
    pub available_bytes: u64,
    /// Used memory in bytes
    pub used_bytes: u64,
    /// Memory usage percentage
    pub usage_percent: f32,
}

/// GPU information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuInfo {
    /// GPU index
    pub index: usize,
    /// GPU model
    pub model: String,
    /// GPU vendor
    pub vendor: String,
    /// Total memory in bytes
    pub total_memory_bytes: u64,
    /// Available memory in bytes
    pub available_memory_bytes: u64,
    /// Used memory in bytes
    pub used_memory_bytes: u64,
    /// Memory usage percentage
    pub memory_usage_percent: f32,
    /// GPU usage percentage
    pub usage_percent: f32,
    /// GPU temperature in Celsius
    pub temperature_celsius: f32,
    /// GPU driver version
    pub driver_version: String,
    /// GPU compute capability
    pub compute_capability: Option<String>,
}

/// Disk information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiskInfo {
    /// Total disk space in bytes
    pub total_bytes: u64,
    /// Available disk space in bytes
    pub available_bytes: u64,
    /// Used disk space in bytes
    pub used_bytes: u64,
    /// Disk usage percentage
    pub usage_percent: f32,
    /// Disk read rate in bytes per second
    pub read_bytes_per_second: u64,
    /// Disk write rate in bytes per second
    pub write_bytes_per_second: u64,
}

/// Network information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkInfo {
    /// Primary interface name
    pub primary_interface: String,
    /// IP address
    pub ip_address: String,
    /// Connection type
    pub connection_type: String,
    /// Network receive rate in bytes per second
    pub receive_bytes_per_second: u64,
    /// Network transmit rate in bytes per second
    pub transmit_bytes_per_second: u64,
}

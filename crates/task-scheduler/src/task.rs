//! Task definition and management
//!
//! This module provides functionality for defining and managing tasks,
//! including task creation, validation, and lifecycle management.

use std::sync::Arc;
use anyhow::Result;
use tracing::{info, warn, error, debug, trace};
use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use uuid::Uuid;
use chrono::{DateTime, Utc};

use common::error::Error;
use common::models::{Task, TaskId, TaskType, TaskStatus, TaskPriority, ResourceAllocation};
use common::types::ResourceRequirements;
use config::ConfigManager;

/// Task definition with all required information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskDefinition {
    /// Task name
    pub name: String,
    
    /// Task description
    pub description: String,
    
    /// Task type
    pub task_type: TaskType,
    
    /// Task priority
    pub priority: TaskPriority,
    
    /// Task parameters
    pub parameters: HashMap<String, String>,
    
    /// Model ID (if applicable)
    pub model_id: Option<String>,
    
    /// Input data (if applicable)
    pub input_data: Option<String>,
    
    /// Output destination (if applicable)
    pub output_destination: Option<String>,
    
    /// Maximum execution time in seconds (0 for unlimited)
    pub max_execution_time: u64,
    
    /// Resource requirements
    pub resource_requirements: Option<ResourceRequirements>,
    
    /// Dependencies (task IDs that must complete before this task)
    pub dependencies: Vec<TaskId>,
    
    /// Tags for categorization
    pub tags: Vec<String>,
    
    /// User ID who submitted the task
    pub user_id: String,
}

/// Task factory for creating and validating tasks
pub struct TaskFactory {
    /// Configuration manager
    config_manager: Arc<ConfigManager>,
    
    /// Default resource requirements by task type
    default_requirements: HashMap<TaskType, ResourceRequirements>,
}

impl TaskFactory {
    /// Creates a new task factory
    pub fn new(config_manager: Arc<ConfigManager>) -> Result<Self> {
        let mut factory = Self {
            config_manager,
            default_requirements: HashMap::new(),
        };
        
        // Initialize default resource requirements
        factory.initialize_default_requirements()?;
        
        Ok(factory)
    }
    
    /// Initializes default resource requirements
    fn initialize_default_requirements(&mut self) -> Result<()> {
        // Default requirements for inference tasks
        self.default_requirements.insert(
            TaskType::Inference,
            ResourceRequirements {
                min_cpu_cores: 1,
                min_memory_bytes: 1024 * 1024 * 1024, // 1 GB
                min_gpu_memory_bytes: Some(1024 * 1024 * 1024), // 1 GB
                min_disk_bytes: 100 * 1024 * 1024, // 100 MB
                requires_gpu: true,
                required_cpu_features: vec!["sse4.1".to_string(), "avx".to_string()],
                required_gpu_features: vec![],
            },
        );
        
        // Default requirements for fine-tuning tasks
        self.default_requirements.insert(
            TaskType::FineTuning,
            ResourceRequirements {
                min_cpu_cores: 4,
                min_memory_bytes: 8 * 1024 * 1024 * 1024, // 8 GB
                min_gpu_memory_bytes: Some(8 * 1024 * 1024 * 1024), // 8 GB
                min_disk_bytes: 10 * 1024 * 1024 * 1024, // 10 GB
                requires_gpu: true,
                required_cpu_features: vec!["sse4.1".to_string(), "avx".to_string(), "avx2".to_string()],
                required_gpu_features: vec![],
            },
        );
        
        // Default requirements for evaluation tasks
        self.default_requirements.insert(
            TaskType::Evaluation,
            ResourceRequirements {
                min_cpu_cores: 2,
                min_memory_bytes: 4 * 1024 * 1024 * 1024, // 4 GB
                min_gpu_memory_bytes: Some(4 * 1024 * 1024 * 1024), // 4 GB
                min_disk_bytes: 1 * 1024 * 1024 * 1024, // 1 GB
                requires_gpu: true,
                required_cpu_features: vec!["sse4.1".to_string(), "avx".to_string()],
                required_gpu_features: vec![],
            },
        );
        
        // Default requirements for export tasks
        self.default_requirements.insert(
            TaskType::Export,
            ResourceRequirements {
                min_cpu_cores: 2,
                min_memory_bytes: 4 * 1024 * 1024 * 1024, // 4 GB
                min_gpu_memory_bytes: None,
                min_disk_bytes: 10 * 1024 * 1024 * 1024, // 10 GB
                requires_gpu: false,
                required_cpu_features: vec![],
                required_gpu_features: vec![],
            },
        );
        
        Ok(())
    }
    
    /// Creates a new task from a task definition
    pub fn create_task(&self, definition: TaskDefinition) -> Result<Task> {
        // Validate task definition
        self.validate_task_definition(&definition)?;
        
        // Generate task ID
        let task_id = Uuid::new_v4().to_string();
        
        // Get resource requirements
        let resource_requirements = match &definition.resource_requirements {
            Some(req) => req.clone(),
            None => self.get_default_requirements(&definition.task_type)?,
        };
        
        // Create task
        let task = Task {
            id: task_id,
            name: definition.name,
            description: definition.description,
            task_type: definition.task_type,
            status: TaskStatus::Created,
            parameters: definition.parameters,
            model_id: definition.model_id,
            input_data: definition.input_data,
            output_destination: definition.output_destination,
            resource_requirements,
            dependencies: definition.dependencies,
            tags: definition.tags,
            user_id: definition.user_id,
            created_at: Utc::now(),
            started_at: None,
            completed_at: None,
            error: None,
            progress: 0.0,
            result: None,
            max_execution_time: definition.max_execution_time,
        };
        
        Ok(task)
    }
    
    /// Validates a task definition
    fn validate_task_definition(&self, definition: &TaskDefinition) -> Result<()> {
        // Check name
        if definition.name.is_empty() {
            return Err(Error::InvalidArgument("Task name cannot be empty".to_string()).into());
        }
        
        // Check model ID for inference tasks
        if matches!(definition.task_type, TaskType::Inference) && definition.model_id.is_none() {
            return Err(Error::InvalidArgument("Model ID is required for inference tasks".to_string()).into());
        }
        
        // Check input data for inference and fine-tuning tasks
        if (matches!(definition.task_type, TaskType::Inference) || 
            matches!(definition.task_type, TaskType::FineTuning)) && 
           definition.input_data.is_none() {
            return Err(Error::InvalidArgument("Input data is required for inference and fine-tuning tasks".to_string()).into());
        }
        
        // Check resource requirements if provided
        if let Some(req) = &definition.resource_requirements {
            // Validate CPU cores
            if req.min_cpu_cores == 0 {
                return Err(Error::InvalidArgument("Minimum CPU cores must be greater than 0".to_string()).into());
            }
            
            // Validate memory
            if req.min_memory_bytes == 0 {
                return Err(Error::InvalidArgument("Minimum memory must be greater than 0".to_string()).into());
            }
            
            // Validate GPU memory if required
            if req.requires_gpu && req.min_gpu_memory_bytes.is_none() {
                return Err(Error::InvalidArgument("GPU memory must be specified if GPU is required".to_string()).into());
            }
        }
        
        Ok(())
    }
    
    /// Gets default resource requirements for a task type
    fn get_default_requirements(&self, task_type: &TaskType) -> Result<ResourceRequirements> {
        match self.default_requirements.get(task_type) {
            Some(req) => Ok(req.clone()),
            None => {
                // For custom task types, use conservative defaults
                Ok(ResourceRequirements {
                    min_cpu_cores: 2,
                    min_memory_bytes: 2 * 1024 * 1024 * 1024, // 2 GB
                    min_gpu_memory_bytes: None,
                    min_disk_bytes: 1 * 1024 * 1024 * 1024, // 1 GB
                    requires_gpu: false,
                    required_cpu_features: vec![],
                    required_gpu_features: vec![],
                })
            }
        }
    }
    
    /// Updates default resource requirements for a task type
    pub fn update_default_requirements(&mut self, task_type: TaskType, requirements: ResourceRequirements) {
        self.default_requirements.insert(task_type, requirements);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    // These tests would require mocking the dependencies
    // For now, we just have placeholder tests
    
    #[test]
    fn test_task_factory() {
        // Placeholder test
        assert!(true);
    }
}

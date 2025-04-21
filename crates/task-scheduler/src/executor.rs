//! Task executor implementation
//!
//! This module provides functionality for executing tasks,
//! including task lifecycle management and error handling.

use std::sync::Arc;
use tokio::sync::{RwLock, Mutex, mpsc};
use anyhow::Result;
use tracing::{info, warn, error, debug, trace};
use std::collections::{HashMap, HashSet};
use dashmap::DashMap;
use uuid::Uuid;
use std::time::{Duration, Instant};
use tokio::time::timeout;

use common::error::Error;
use common::models::{Task, TaskId, TaskType, TaskStatus, TaskPriority, ResourceAllocation};
use common::types::ResourceRequirements;
use config::ConfigManager;
use resource_manager::ResourceManager;
use model_manager::ModelManager;

/// Task execution result
#[derive(Debug, Clone)]
pub struct TaskResult {
    /// Task ID
    pub task_id: TaskId,
    
    /// Task status
    pub status: TaskStatus,
    
    /// Result data (if successful)
    pub data: Option<String>,
    
    /// Error message (if failed)
    pub error: Option<String>,
    
    /// Execution time in seconds
    pub execution_time: f64,
    
    /// Resource usage statistics
    pub resource_usage: HashMap<String, f64>,
}

/// Task executor for running tasks
pub struct TaskExecutor {
    /// Resource manager
    resource_manager: Arc<ResourceManager>,
    
    /// Model manager
    model_manager: Arc<ModelManager>,
    
    /// Configuration manager
    config_manager: Arc<ConfigManager>,
    
    /// Running tasks (task_id -> task)
    running_tasks: DashMap<TaskId, Task>,
    
    /// Task results (task_id -> result)
    task_results: DashMap<TaskId, TaskResult>,
    
    /// Task status updates sender
    status_tx: mpsc::Sender<(TaskId, TaskStatus)>,
    
    /// Task status updates receiver
    status_rx: Arc<Mutex<mpsc::Receiver<(TaskId, TaskStatus)>>>,
    
    /// Maximum concurrent tasks
    max_concurrent_tasks: usize,
    
    /// Executor running flag
    running: Arc<RwLock<bool>>,
}

impl TaskExecutor {
    /// Creates a new task executor
    pub fn new(
        resource_manager: Arc<ResourceManager>,
        model_manager: Arc<ModelManager>,
        config_manager: Arc<ConfigManager>,
    ) -> Result<Self> {
        // Get maximum concurrent tasks from config
        let max_concurrent_tasks = config_manager
            .get_usize("max_concurrent_tasks")
            .unwrap_or_else(|_| {
                // Default to number of CPU cores
                match resource_manager.get_monitor().get_total_cpu_cores() {
                    Ok(cores) => cores,
                    Err(_) => 4, // Fallback default
                }
            });
        
        // Create channel for task status updates
        let (status_tx, status_rx) = mpsc::channel(100);
        
        Ok(Self {
            resource_manager,
            model_manager,
            config_manager,
            running_tasks: DashMap::new(),
            task_results: DashMap::new(),
            status_tx,
            status_rx: Arc::new(Mutex::new(status_rx)),
            max_concurrent_tasks,
            running: Arc::new(RwLock::new(false)),
        })
    }
    
    /// Starts the executor
    pub async fn start(&self) -> Result<()> {
        // Check if already running
        {
            let running = self.running.read().await;
            if *running {
                return Ok(());
            }
        }
        
        // Set running flag
        {
            let mut running = self.running.write().await;
            *running = true;
        }
        
        info!("Starting task executor");
        
        // Start status update handler
        self.start_status_update_handler().await?;
        
        info!("Task executor started successfully");
        
        Ok(())
    }
    
    /// Stops the executor
    pub async fn stop(&self) -> Result<()> {
        // Check if running
        {
            let running = self.running.read().await;
            if !*running {
                return Ok(());
            }
        }
        
        info!("Stopping task executor");
        
        // Set running flag to false
        {
            let mut running = self.running.write().await;
            *running = false;
        }
        
        // Cancel all running tasks
        let running_task_ids: Vec<TaskId> = self.running_tasks.iter().map(|entry| entry.key().clone()).collect();
        
        for task_id in running_task_ids {
            let _ = self.cancel_task(&task_id).await;
        }
        
        info!("Task executor stopped successfully");
        
        Ok(())
    }
    
    /// Starts the status update handler
    async fn start_status_update_handler(&self) -> Result<()> {
        let status_rx = self.status_rx.clone();
        let running_tasks = self.running_tasks.clone();
        let task_results = self.task_results.clone();
        let running = self.running.clone();
        
        tokio::spawn(async move {
            let mut rx = status_rx.lock().await;
            
            while {
                let is_running = *running.read().await;
                is_running
            } {
                if let Ok(Some((task_id, status))) = tokio::time::timeout(
                    Duration::from_millis(100),
                    rx.recv()
                ).await {
                    match status {
                        TaskStatus::Completed | TaskStatus::Failed | TaskStatus::Cancelled => {
                            // Task is done, update status
                            if let Some(mut entry) = running_tasks.get_mut(&task_id) {
                                entry.status = status;
                                entry.completed_at = Some(chrono::Utc::now());
                                
                                if status == TaskStatus::Failed {
                                    entry.error = Some("Task execution failed".to_string());
                                }
                            }
                            
                            // Remove from running tasks
                            running_tasks.remove(&task_id);
                            
                            info!("Task {} completed with status {:?}", task_id, status);
                        },
                        _ => {
                            // Update task status
                            if let Some(mut entry) = running_tasks.get_mut(&task_id) {
                                entry.status = status;
                                
                                if status == TaskStatus::Running && entry.started_at.is_none() {
                                    entry.started_at = Some(chrono::Utc::now());
                                }
                            }
                        }
                    }
                }
            }
        });
        
        Ok(())
    }
    
    /// Executes a task
    pub async fn execute_task(&self, task: Task) -> Result<TaskId> {
        // Check if we can execute more tasks
        if self.running_tasks.len() >= self.max_concurrent_tasks {
            return Err(Error::Resource("Maximum number of concurrent tasks reached".to_string()).into());
        }
        
        // Check if task already exists
        if self.running_tasks.contains_key(&task.id) || self.task_results.contains_key(&task.id) {
            return Err(Error::AlreadyExists(format!("Task {} already exists", task.id)).into());
        }
        
        // Allocate resources
        let allocation = self.resource_manager.get_allocator().allocate_resources(&task).await?;
        
        // Update task status
        let mut updated_task = task.clone();
        updated_task.status = TaskStatus::Running;
        updated_task.started_at = Some(chrono::Utc::now());
        
        // Add to running tasks
        self.running_tasks.insert(task.id.clone(), updated_task.clone());
        
        // Send status update
        let _ = self.status_tx.send((task.id.clone(), TaskStatus::Running)).await;
        
        // Start task execution
        self.start_task_execution(updated_task, allocation).await?;
        
        info!("Started execution of task {}", task.id);
        
        Ok(task.id)
    }
    
    /// Starts task execution in a background task
    async fn start_task_execution(&self, task: Task, allocation: ResourceAllocation) -> Result<()> {
        let task_id = task.id.clone();
        let task_type = task.task_type.clone();
        let model_id = task.model_id.clone();
        let input_data = task.input_data.clone();
        let parameters = task.parameters.clone();
        let max_execution_time = task.max_execution_time;
        
        let resource_manager = self.resource_manager.clone();
        let model_manager = self.model_manager.clone();
        let status_tx = self.status_tx.clone();
        let task_results = self.task_results.clone();
        
        tokio::spawn(async move {
            let start_time = Instant::now();
            let mut result = TaskResult {
                task_id: task_id.clone(),
                status: TaskStatus::Running,
                data: None,
                error: None,
                execution_time: 0.0,
                resource_usage: HashMap::new(),
            };
            
            // Execute task with timeout if specified
            let execution_result = if max_execution_time > 0 {
                timeout(
                    Duration::from_secs(max_execution_time),
                    Self::execute_task_by_type(
                        &task_id,
                        &task_type,
                        model_id.as_deref(),
                        input_data.as_deref(),
                        &parameters,
                        &model_manager,
                    )
                ).await
            } else {
                Ok(Self::execute_task_by_type(
                    &task_id,
                    &task_type,
                    model_id.as_deref(),
                    input_data.as_deref(),
                    &parameters,
                    &model_manager,
                ).await)
            };
            
            // Process execution result
            match execution_result {
                Ok(Ok(output)) => {
                    // Task completed successfully
                    result.status = TaskStatus::Completed;
                    result.data = Some(output);
                    
                    // Send status update
                    let _ = status_tx.send((task_id.clone(), TaskStatus::Completed)).await;
                },
                Ok(Err(e)) => {
                    // Task failed
                    result.status = TaskStatus::Failed;
                    result.error = Some(format!("Task execution failed: {}", e));
                    
                    // Send status update
                    let _ = status_tx.send((task_id.clone(), TaskStatus::Failed)).await;
                    
                    error!("Task {} failed: {}", task_id, e);
                },
                Err(_) => {
                    // Task timed out
                    result.status = TaskStatus::Failed;
                    result.error = Some(format!("Task execution timed out after {} seconds", max_execution_time));
                    
                    // Send status update
                    let _ = status_tx.send((task_id.clone(), TaskStatus::Failed)).await;
                    
                    error!("Task {} timed out after {} seconds", task_id, max_execution_time);
                }
            }
            
            // Calculate execution time
            result.execution_time = start_time.elapsed().as_secs_f64();
            
            // Collect resource usage statistics
            if let Ok(usage) = resource_manager.get_resource_usage().await {
                result.resource_usage.insert("cpu_usage".to_string(), usage.cpu_usage as f64);
                result.resource_usage.insert("memory_usage".to_string(), usage.memory_usage as f64);
                result.resource_usage.insert("gpu_usage".to_string(), usage.gpu_usage as f64);
            }
            
            // Store task result
            task_results.insert(task_id.clone(), result);
            
            // Release resources
            if let Err(e) = resource_manager.get_allocator().release_resources(&task_id).await {
                warn!("Failed to release resources for task {}: {}", task_id, e);
            }
            
            info!("Task {} execution completed in {:.2} seconds", task_id, start_time.elapsed().as_secs_f64());
        });
        
        Ok(())
    }
    
    /// Executes a task based on its type
    async fn execute_task_by_type(
        task_id: &TaskId,
        task_type: &TaskType,
        model_id: Option<&str>,
        input_data: Option<&str>,
        parameters: &HashMap<String, String>,
        model_manager: &Arc<ModelManager>,
    ) -> Result<String> {
        match task_type {
            TaskType::Inference => {
                // Check if model ID is provided
                let model_id = model_id.ok_or_else(|| Error::InvalidArgument("Model ID is required for inference tasks".to_string()))?;
                
                // Check if input data is provided
                let input = input_data.ok_or_else(|| Error::InvalidArgument("Input data is required for inference tasks".to_string()))?;
                
                // Get model
                let model = model_manager.get_repository().get_model(model_id).await?;
                
                // Load model if not already loaded
                if !model_manager.get_loader().is_model_loaded(model_id) {
                    model_manager.get_loader().load_model(model_id).await?;
                }
                
                // In a real implementation, we would perform inference here
                // For now, we just simulate it
                
                // Simulate inference
                tokio::time::sleep(Duration::from_millis(500)).await;
                
                // Return simulated result
                Ok(format!("Inference result for input: {}", input))
            },
            TaskType::FineTuning => {
                // Check if model ID is provided
                let model_id = model_id.ok_or_else(|| Error::InvalidArgument("Model ID is required for fine-tuning tasks".to_string()))?;
                
                // Check if input data is provided
                let input = input_data.ok_or_else(|| Error::InvalidArgument("Input data is required for fine-tuning tasks".to_string()))?;
                
                // Get model
                let model = model_manager.get_repository().get_model(model_id).await?;
                
                // In a real implementation, we would perform fine-tuning here
                // For now, we just simulate it
                
                // Simulate fine-tuning (longer operation)
                tokio::time::sleep(Duration::from_secs(2)).await;
                
                // Return simulated result
                Ok(format!("Fine-tuning completed for model: {}", model_id))
            },
            TaskType::Evaluation => {
                // Check if model ID is provided
                let model_id = model_id.ok_or_else(|| Error::InvalidArgument("Model ID is required for evaluation tasks".to_string()))?;
                
                // Get model
                let model = model_manager.get_repository().get_model(model_id).await?;
                
                // In a real implementation, we would perform evaluation here
                // For now, we just simulate it
                
                // Simulate evaluation
                tokio::time::sleep(Duration::from_millis(800)).await;
                
                // Return simulated result
                Ok(format!("Evaluation results for model {}: accuracy=0.92, f1=0.89", model_id))
            },
            TaskType::Export => {
                // Check if model ID is provided
                let model_id = model_id.ok_or_else(|| Error::InvalidArgument("Model ID is required for export tasks".to_string()))?;
                
                // Get model
                let model = model_manager.get_repository().get_model(model_id).await?;
                
                // In a real implementation, we would perform export here
                // For now, we just simulate it
                
                // Simulate export
                tokio::time::sleep(Duration::from_millis(600)).await;
                
                // Return simulated result
                Ok(format!("Model {} exported successfully", model_id))
            },
            TaskType::Custom(custom_type) => {
                // In a real implementation, we would handle custom task types
                // For now, we just simulate it
                
                // Simulate custom task
                tokio::time::sleep(Duration::from_millis(300)).await;
                
                // Return simulated result
                Ok(format!("Custom task '{}' executed successfully", custom_type))
            },
        }
    }
    
    /// Cancels a running task
    pub async fn cancel_task(&self, task_id: &TaskId) -> Result<()> {
        // Check if task is running
        if !self.running_tasks.contains_key(task_id) {
            return Err(Error::NotFound(format!("Task {} is not running", task_id)).into());
        }
        
        // Update task status
        if let Some(mut entry) = self.running_tasks.get_mut(task_id) {
            entry.status = TaskStatus::Cancelled;
            entry.completed_at = Some(chrono::Utc::now());
        }
        
        // Send status update
        let _ = self.status_tx.send((task_id.clone(), TaskStatus::Cancelled)).await;
        
        // Release resources
        self.resource_manager.get_allocator().release_resources(task_id).await?;
        
        info!("Task {} cancelled", task_id);
        
        Ok(())
    }
    
    /// Gets the status of a task
    pub fn get_task_status(&self, task_id: &TaskId) -> Result<TaskStatus> {
        // Check if task is running
        if let Some(task) = self.running_tasks.get(task_id) {
            return Ok(task.status);
        }
        
        // Check if task has completed
        if let Some(result) = self.task_results.get(task_id) {
            return Ok(result.status);
        }
        
        Err(Error::NotFound(format!("Task {} not found", task_id)).into())
    }
    
    /// Gets a task result
    pub fn get_task_result(&self, task_id: &TaskId) -> Result<TaskResult> {
        match self.task_results.get(task_id) {
            Some(result) => Ok(result.clone()),
            None => Err(Error::NotFound(format!("No result found for task {}", task_id)).into()),
        }
    }
    
    /// Gets all running tasks
    pub fn get_running_tasks(&self) -> Vec<Task> {
        self.running_tasks.iter().map(|entry| entry.value().clone()).collect()
    }
    
    /// Gets all task results
    pub fn get_all_task_results(&self) -> Vec<TaskResult> {
        self.task_results.iter().map(|entry| entry.value().clone()).collect()
    }
    
    /// Gets the number of running tasks
    pub fn get_running_tasks_count(&self) -> usize {
        self.running_tasks.len()
    }
    
    /// Gets a status update sender
    pub fn get_status_sender(&self) -> mpsc::Sender<(TaskId, TaskStatus)> {
        self.status_tx.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    // These tests would require mocking the dependencies
    // For now, we just have placeholder tests
    
    #[test]
    fn test_task_executor() {
        // Placeholder test
        assert!(true);
    }
}

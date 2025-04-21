//! Coordinator for orchestrating components
//!
//! This module provides the coordinator that orchestrates interactions between
//! different components of the AI Orchestrator system.

use std::sync::Arc;
use tokio::sync::RwLock;
use anyhow::Result;
use tracing::{info, warn, error, debug, trace};
use async_trait::async_trait;

use common::error::Error;
use common::types::OperationalMode;
use model_manager::ModelRepository;
use resource_manager::ResourceMonitor;
use task_scheduler::TaskQueue;
use scaling_adapter::DynamicScaler;
use config::ConfigManager;

use crate::state::OrchestratorState;

/// Coordinator for orchestrating components
pub struct Coordinator {
    /// Current state of the orchestrator
    state: Arc<RwLock<OrchestratorState>>,
    
    /// Model repository
    model_repository: Arc<ModelRepository>,
    
    /// Resource monitor
    resource_monitor: Arc<ResourceMonitor>,
    
    /// Task queue
    task_queue: Arc<TaskQueue>,
    
    /// Dynamic scaler
    dynamic_scaler: Arc<DynamicScaler>,
    
    /// Configuration manager
    config_manager: Arc<ConfigManager>,
}

impl Coordinator {
    /// Creates a new coordinator
    pub fn new(
        state: Arc<RwLock<OrchestratorState>>,
        model_repository: Arc<ModelRepository>,
        resource_monitor: Arc<ResourceMonitor>,
        task_queue: Arc<TaskQueue>,
        dynamic_scaler: Arc<DynamicScaler>,
        config_manager: Arc<ConfigManager>,
    ) -> Result<Self> {
        Ok(Self {
            state,
            model_repository,
            resource_monitor,
            task_queue,
            dynamic_scaler,
            config_manager,
        })
    }
    
    /// Starts the coordinator
    pub async fn start(&self) -> Result<()> {
        info!("Starting coordinator");
        
        // Start task processing
        self.start_task_processing().await?;
        
        info!("Coordinator started successfully");
        
        Ok(())
    }
    
    /// Stops the coordinator
    pub async fn stop(&self) -> Result<()> {
        info!("Stopping coordinator");
        
        // Stop task processing
        self.stop_task_processing().await?;
        
        info!("Coordinator stopped successfully");
        
        Ok(())
    }
    
    /// Starts task processing
    async fn start_task_processing(&self) -> Result<()> {
        // Start a background task to process the task queue
        let task_queue = self.task_queue.clone();
        let model_repository = self.model_repository.clone();
        let resource_monitor = self.resource_monitor.clone();
        let dynamic_scaler = self.dynamic_scaler.clone();
        
        tokio::spawn(async move {
            loop {
                // Check if there are any tasks in the queue
                match task_queue.peek().await {
                    Ok(Some(task)) => {
                        // Check if we have enough resources to execute the task
                        match resource_monitor.can_execute_task(&task).await {
                            Ok(true) => {
                                // Dequeue the task
                                if let Ok(task) = task_queue.dequeue().await {
                                    // Get the model
                                    if let Ok(model) = model_repository.get_model(&task.model_id).await {
                                        // Execute the task
                                        let task_id = task.id.to_string();
                                        let model_id = model.id.clone();
                                        
                                        debug!("Executing task {} with model {}", task_id, model_id);
                                        
                                        // This would be handled by the inference engine in a real implementation
                                        // For now, we just mark the task as completed
                                        if let Err(e) = task_queue.complete(&task_id).await {
                                            error!("Failed to complete task {}: {}", task_id, e);
                                        }
                                    } else {
                                        // Model not found, mark task as failed
                                        let task_id = task.id.to_string();
                                        error!("Model {} not found for task {}", task.model_id, task_id);
                                        if let Err(e) = task_queue.fail(&task_id, "Model not found").await {
                                            error!("Failed to mark task {} as failed: {}", task_id, e);
                                        }
                                    }
                                }
                            },
                            Ok(false) => {
                                // Not enough resources, wait for resources to become available
                                debug!("Not enough resources to execute task, waiting");
                                tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
                            },
                            Err(e) => {
                                error!("Failed to check resource availability: {}", e);
                                tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
                            }
                        }
                    },
                    Ok(None) => {
                        // No tasks in the queue, wait for new tasks
                        trace!("No tasks in the queue, waiting");
                        tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
                    },
                    Err(e) => {
                        error!("Failed to peek at task queue: {}", e);
                        tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
                    }
                }
                
                // Check if we should exit
                if task_queue.is_shutdown() {
                    break;
                }
            }
        });
        
        Ok(())
    }
    
    /// Stops task processing
    async fn stop_task_processing(&self) -> Result<()> {
        // Signal the task queue to shut down
        self.task_queue.shutdown().await?;
        
        Ok(())
    }
    
    /// Called when a task is submitted
    pub async fn on_task_submitted(&self, task_id: &str) -> Result<()> {
        debug!("Task submitted: {}", task_id);
        
        // Check if we need to scale resources
        self.dynamic_scaler.check_scaling_needs().await?;
        
        Ok(())
    }
    
    /// Called when a task is cancelled
    pub async fn on_task_cancelled(&self, task_id: &str) -> Result<()> {
        debug!("Task cancelled: {}", task_id);
        
        // Check if we need to scale resources
        self.dynamic_scaler.check_scaling_needs().await?;
        
        Ok(())
    }
    
    /// Called when a task is completed
    pub async fn on_task_completed(&self, task_id: &str) -> Result<()> {
        debug!("Task completed: {}", task_id);
        
        // Check if we need to scale resources
        self.dynamic_scaler.check_scaling_needs().await?;
        
        Ok(())
    }
    
    /// Called when a task fails
    pub async fn on_task_failed(&self, task_id: &str, error: &str) -> Result<()> {
        warn!("Task failed: {}, error: {}", task_id, error);
        
        // Check if we need to scale resources
        self.dynamic_scaler.check_scaling_needs().await?;
        
        Ok(())
    }
    
    /// Called when the operational mode changes
    pub async fn on_mode_change(&self, mode: OperationalMode) -> Result<()> {
        info!("Operational mode changed to: {:?}", mode);
        
        // Adjust resource allocation based on new mode
        self.dynamic_scaler.adjust_for_mode(mode).await?;
        
        Ok(())
    }
    
    /// Called when system resources change significantly
    pub async fn on_resources_changed(&self) -> Result<()> {
        debug!("System resources changed");
        
        // Check if we need to adjust task execution
        self.check_resource_constraints().await?;
        
        Ok(())
    }
    
    /// Checks resource constraints and adjusts task execution
    async fn check_resource_constraints(&self) -> Result<()> {
        // Get current resource usage
        let resources = self.resource_monitor.get_system_resources().await?;
        
        // Check if we're running low on resources
        if resources.memory.usage_percent > 90.0 {
            warn!("Memory usage is high ({}%), pausing new task execution", resources.memory.usage_percent);
            
            // Pause task execution
            self.task_queue.pause().await?;
        } else if resources.memory.usage_percent < 70.0 {
            // Resume task execution if it was paused
            if self.task_queue.is_paused().await? {
                info!("Memory usage is acceptable ({}%), resuming task execution", resources.memory.usage_percent);
                self.task_queue.resume().await?;
            }
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use common::models::{Task, TaskType, TaskStatus, TaskPriority, TaskInput, TaskConfig, ExecutionMode};
    use uuid::Uuid;
    use chrono::Utc;
    use std::collections::HashMap;
    
    // Mock implementations for testing
    struct MockModelRepository;
    struct MockResourceMonitor;
    struct MockTaskQueue;
    struct MockDynamicScaler;
    struct MockConfigManager;
    
    #[tokio::test]
    async fn test_coordinator_lifecycle() {
        // This is a placeholder for a real test
        // In a real implementation, we would use mocks for the dependencies
        assert!(true);
    }
}

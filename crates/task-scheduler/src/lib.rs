//! Main module for the task scheduler
//!
//! This module provides the main entry point for the task scheduler,
//! integrating task creation, queuing, and execution.

mod task;
mod queue;
mod executor;

pub use task::{TaskDefinition, TaskFactory};
pub use queue::TaskQueueManager;
pub use executor::{TaskExecutor, TaskResult};

use std::sync::Arc;
use tokio::sync::{RwLock, mpsc};
use anyhow::Result;
use tracing::{info, warn, error, debug, trace};
use std::collections::HashMap;

use common::error::Error;
use common::models::{Task, TaskId, TaskType, TaskStatus, TaskPriority};
use resource_manager::ResourceManager;
use model_manager::ModelManager;
use config::ConfigManager;

/// Main task scheduler that integrates all task scheduling components
pub struct TaskScheduler {
    /// Task factory
    task_factory: Arc<TaskFactory>,
    
    /// Task queue manager
    queue_manager: Arc<TaskQueueManager>,
    
    /// Task executor
    executor: Arc<TaskExecutor>,
    
    /// Resource manager
    resource_manager: Arc<ResourceManager>,
    
    /// Model manager
    model_manager: Arc<ModelManager>,
    
    /// Configuration manager
    config_manager: Arc<ConfigManager>,
    
    /// Scheduler running flag
    running: Arc<RwLock<bool>>,
    
    /// Scheduler interval
    scheduler_interval: std::time::Duration,
}

impl TaskScheduler {
    /// Creates a new task scheduler
    pub fn new(
        resource_manager: Arc<ResourceManager>,
        model_manager: Arc<ModelManager>,
        config_manager: Arc<ConfigManager>,
    ) -> Result<Self> {
        // Create task factory
        let task_factory = Arc::new(TaskFactory::new(config_manager.clone())?);
        
        // Create task queue manager
        let queue_manager = Arc::new(TaskQueueManager::new(config_manager.clone())?);
        
        // Create task executor
        let executor = Arc::new(TaskExecutor::new(
            resource_manager.clone(),
            model_manager.clone(),
            config_manager.clone(),
        )?);
        
        // Get scheduler interval from config
        let scheduler_interval = config_manager
            .get_duration("scheduler_interval")
            .unwrap_or_else(|_| std::time::Duration::from_millis(100));
        
        Ok(Self {
            task_factory,
            queue_manager,
            executor,
            resource_manager,
            model_manager,
            config_manager,
            running: Arc::new(RwLock::new(false)),
            scheduler_interval,
        })
    }
    
    /// Starts the task scheduler
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
        
        info!("Starting task scheduler");
        
        // Start executor
        self.executor.start().await?;
        
        // Start scheduler loop
        self.start_scheduler_loop().await?;
        
        info!("Task scheduler started successfully");
        
        Ok(())
    }
    
    /// Stops the task scheduler
    pub async fn stop(&self) -> Result<()> {
        // Check if running
        {
            let running = self.running.read().await;
            if !*running {
                return Ok(());
            }
        }
        
        info!("Stopping task scheduler");
        
        // Set running flag to false
        {
            let mut running = self.running.write().await;
            *running = false;
        }
        
        // Stop executor
        self.executor.stop().await?;
        
        info!("Task scheduler stopped successfully");
        
        Ok(())
    }
    
    /// Starts the scheduler loop
    async fn start_scheduler_loop(&self) -> Result<()> {
        let queue_manager = self.queue_manager.clone();
        let executor = self.executor.clone();
        let running = self.running.clone();
        let scheduler_interval = self.scheduler_interval;
        
        tokio::spawn(async move {
            while {
                let is_running = *running.read().await;
                is_running
            } {
                // Check if we can execute more tasks
                if executor.get_running_tasks_count() < executor.max_concurrent_tasks {
                    // Get next task from queue
                    if let Some(task) = queue_manager.dequeue_task().await {
                        // Execute task
                        match executor.execute_task(task.clone()).await {
                            Ok(_) => {
                                debug!("Started execution of task {}", task.id);
                            },
                            Err(e) => {
                                warn!("Failed to execute task {}: {}", task.id, e);
                                
                                // Notify that task has failed
                                let _ = executor.get_status_sender().send((task.id.clone(), TaskStatus::Failed)).await;
                            }
                        }
                    }
                }
                
                // Sleep for scheduler interval
                tokio::time::sleep(scheduler_interval).await;
            }
        });
        
        Ok(())
    }
    
    /// Submits a task for execution
    pub async fn submit_task(&self, definition: TaskDefinition) -> Result<TaskId> {
        // Create task
        let task = self.task_factory.create_task(definition)?;
        
        // Enqueue task
        self.queue_manager.enqueue_task(task.clone(), task.priority()).await?;
        
        info!("Task {} submitted successfully", task.id);
        
        Ok(task.id)
    }
    
    /// Cancels a task
    pub async fn cancel_task(&self, task_id: &TaskId) -> Result<()> {
        // Check if task is in queue
        if self.queue_manager.is_task_queued(task_id) {
            // Cancel queued task
            self.queue_manager.cancel_task(task_id).await?;
            
            info!("Cancelled queued task {}", task_id);
            
            return Ok(());
        }
        
        // Check if task is running
        if let Ok(status) = self.executor.get_task_status(task_id) {
            if status == TaskStatus::Running {
                // Cancel running task
                self.executor.cancel_task(task_id).await?;
                
                info!("Cancelled running task {}", task_id);
                
                return Ok(());
            }
        }
        
        Err(Error::NotFound(format!("Task {} not found or not cancellable", task_id)).into())
    }
    
    /// Gets the status of a task
    pub async fn get_task_status(&self, task_id: &TaskId) -> Result<TaskStatus> {
        // Check if task is in queue
        if self.queue_manager.is_task_queued(task_id) {
            return Ok(TaskStatus::Queued);
        }
        
        // Check if task is running or completed
        self.executor.get_task_status(task_id)
    }
    
    /// Gets a task result
    pub fn get_task_result(&self, task_id: &TaskId) -> Result<TaskResult> {
        self.executor.get_task_result(task_id)
    }
    
    /// Gets all tasks
    pub async fn get_all_tasks(&self) -> Result<Vec<Task>> {
        let mut tasks = Vec::new();
        
        // Get queued tasks
        let queued_tasks = self.queue_manager.get_all_queued_tasks().await;
        tasks.extend(queued_tasks);
        
        // Get running tasks
        let running_tasks = self.executor.get_running_tasks();
        tasks.extend(running_tasks);
        
        Ok(tasks)
    }
    
    /// Gets the number of queued tasks
    pub fn get_queue_length(&self) -> Result<usize> {
        self.queue_manager.get_queue_size()
    }
    
    /// Gets the number of running tasks
    pub fn get_running_tasks_count(&self) -> usize {
        self.executor.get_running_tasks_count()
    }
    
    /// Gets the task factory
    pub fn get_task_factory(&self) -> Arc<TaskFactory> {
        self.task_factory.clone()
    }
    
    /// Gets the task queue manager
    pub fn get_queue_manager(&self) -> Arc<TaskQueueManager> {
        self.queue_manager.clone()
    }
    
    /// Gets the task executor
    pub fn get_executor(&self) -> Arc<TaskExecutor> {
        self.executor.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    // These tests would require mocking the dependencies
    // For now, we just have placeholder tests
    
    #[test]
    fn test_task_scheduler() {
        // Placeholder test
        assert!(true);
    }
}

//! Main integration module for the AI orchestrator
//!
//! This module integrates all components of the AI orchestrator
//! and provides the main entry point for the application.

use std::sync::Arc;
use anyhow::Result;
use tracing::{info, warn, error, debug, trace};
use tracing_subscriber::{EnvFilter, fmt};
use std::time::{Duration, Instant};

use common::error::Error;
use config::ConfigManager;
use model_manager::ModelManager;
use resource_manager::ResourceManager;
use task_scheduler::{TaskScheduler, TaskDefinition, TaskFactory};
use performance_optimizations::PerformanceOptimizer;
use common::models::{Task, TaskId, TaskType, TaskStatus, TaskPriority};

/// Main AI orchestrator
pub struct AIOrchestrator {
    /// Configuration manager
    config_manager: Arc<ConfigManager>,
    
    /// Model manager
    model_manager: Arc<ModelManager>,
    
    /// Resource manager
    resource_manager: Arc<ResourceManager>,
    
    /// Task scheduler
    task_scheduler: Arc<TaskScheduler>,
    
    /// Performance optimizer
    performance_optimizer: Arc<PerformanceOptimizer>,
    
    /// Hardware profiler
    hardware_profiler: Arc<hardware_profiler::HardwareCapabilities>,
}

impl AIOrchestrator {
    /// Creates a new AI orchestrator
    pub async fn new() -> Result<Self> {
        // Initialize logging
        Self::init_logging()?;
        
        info!("Initializing AI orchestrator");
        
        // Create configuration manager
        let config_manager = Arc::new(ConfigManager::new()?);
        
        // Create hardware profiler
        let hardware_profiler = Arc::new(hardware_profiler::HardwareCapabilities::new()?);
        
        // Create performance optimizer
        let performance_optimizer = Arc::new(PerformanceOptimizer::new(config_manager.clone()));
        
        // Create model manager
        let model_manager = Arc::new(ModelManager::new(
            config_manager.clone(),
            hardware_profiler.clone(),
        )?);
        
        // Create resource manager
        let resource_manager = Arc::new(ResourceManager::new(
            hardware_profiler.clone(),
            config_manager.clone(),
        )?);
        
        // Create task scheduler
        let task_scheduler = Arc::new(TaskScheduler::new(
            resource_manager.clone(),
            model_manager.clone(),
            config_manager.clone(),
        )?);
        
        // Initialize memory pools
        Self::init_memory_pools(&performance_optimizer)?;
        
        Ok(Self {
            config_manager,
            model_manager,
            resource_manager,
            task_scheduler,
            performance_optimizer,
            hardware_profiler,
        })
    }
    
    /// Initializes logging
    fn init_logging() -> Result<()> {
        // Initialize tracing subscriber
        let filter = EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| EnvFilter::new("info"));
        
        fmt()
            .with_env_filter(filter)
            .with_target(true)
            .init();
        
        Ok(())
    }
    
    /// Initializes memory pools
    fn init_memory_pools(performance_optimizer: &PerformanceOptimizer) -> Result<()> {
        // Create small object pool
        performance_optimizer.create_memory_pool(
            "small_objects",
            performance_optimizations::MemoryPoolConfig {
                block_size: 1024,
                initial_blocks: 100,
                max_blocks: 1000,
                growth_factor: 1.5,
            },
        )?;
        
        // Create medium object pool
        performance_optimizer.create_memory_pool(
            "medium_objects",
            performance_optimizations::MemoryPoolConfig {
                block_size: 64 * 1024,
                initial_blocks: 10,
                max_blocks: 100,
                growth_factor: 1.5,
            },
        )?;
        
        // Create large object pool
        performance_optimizer.create_memory_pool(
            "large_objects",
            performance_optimizations::MemoryPoolConfig {
                block_size: 1024 * 1024,
                initial_blocks: 2,
                max_blocks: 20,
                growth_factor: 2.0,
            },
        )?;
        
        Ok(())
    }
    
    /// Starts the AI orchestrator
    pub async fn start(&self) -> Result<()> {
        info!("Starting AI orchestrator");
        
        // Start resource manager
        self.resource_manager.start().await?;
        
        // Start task scheduler
        self.task_scheduler.start().await?;
        
        info!("AI orchestrator started successfully");
        
        Ok(())
    }
    
    /// Stops the AI orchestrator
    pub async fn stop(&self) -> Result<()> {
        info!("Stopping AI orchestrator");
        
        // Stop task scheduler
        self.task_scheduler.stop().await?;
        
        // Stop resource manager
        self.resource_manager.stop().await?;
        
        info!("AI orchestrator stopped successfully");
        
        Ok(())
    }
    
    /// Submits a task for execution
    pub async fn submit_task(&self, definition: TaskDefinition) -> Result<TaskId> {
        self.task_scheduler.submit_task(definition).await
    }
    
    /// Cancels a task
    pub async fn cancel_task(&self, task_id: &TaskId) -> Result<()> {
        self.task_scheduler.cancel_task(task_id).await
    }
    
    /// Gets the status of a task
    pub async fn get_task_status(&self, task_id: &TaskId) -> Result<TaskStatus> {
        self.task_scheduler.get_task_status(task_id).await
    }
    
    /// Gets a task result
    pub fn get_task_result(&self, task_id: &TaskId) -> Result<task_scheduler::TaskResult> {
        self.task_scheduler.get_task_result(task_id)
    }
    
    /// Gets all tasks
    pub async fn get_all_tasks(&self) -> Result<Vec<Task>> {
        self.task_scheduler.get_all_tasks().await
    }
    
    /// Gets the number of queued tasks
    pub fn get_queue_length(&self) -> Result<usize> {
        self.task_scheduler.get_queue_length()
    }
    
    /// Gets the number of running tasks
    pub fn get_running_tasks_count(&self) -> usize {
        self.task_scheduler.get_running_tasks_count()
    }
    
    /// Gets the current system resources
    pub async fn get_system_resources(&self) -> Result<common::models::SystemResources> {
        self.resource_manager.get_system_resources().await
    }
    
    /// Gets the current resource usage
    pub async fn get_resource_usage(&self) -> Result<resource_manager::ResourceUsage> {
        self.resource_manager.get_resource_usage().await
    }
    
    /// Gets the hardware capabilities
    pub fn get_hardware_capabilities(&self) -> Arc<hardware_profiler::HardwareCapabilities> {
        self.hardware_profiler.clone()
    }
    
    /// Gets the model manager
    pub fn get_model_manager(&self) -> Arc<ModelManager> {
        self.model_manager.clone()
    }
    
    /// Gets the resource manager
    pub fn get_resource_manager(&self) -> Arc<ResourceManager> {
        self.resource_manager.clone()
    }
    
    /// Gets the task scheduler
    pub fn get_task_scheduler(&self) -> Arc<TaskScheduler> {
        self.task_scheduler.clone()
    }
    
    /// Gets the performance optimizer
    pub fn get_performance_optimizer(&self) -> Arc<PerformanceOptimizer> {
        self.performance_optimizer.clone()
    }
    
    /// Gets the configuration manager
    pub fn get_config_manager(&self) -> Arc<ConfigManager> {
        self.config_manager.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    // These tests would require mocking the dependencies
    // For now, we just have placeholder tests
    
    #[test]
    fn test_ai_orchestrator() {
        // Placeholder test
        assert!(true);
    }
}

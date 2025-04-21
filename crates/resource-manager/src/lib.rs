//! Main module for the resource manager
//!
//! This module provides the main entry point for the resource manager,
//! integrating monitoring, allocation, scheduling, and throttling.

mod monitor;
mod allocator;
mod scheduler;
mod throttler;

pub use monitor::{ResourceMonitor, ResourceThresholds, ResourceReservation};
pub use allocator::{ResourceAllocator, ResourceUsage};
pub use scheduler::ResourceScheduler;
pub use throttler::{ResourceThrottler, ThrottlingPolicy};

use std::sync::Arc;
use anyhow::Result;
use tracing::{info, warn, error, debug};

use common::error::Error;
use common::models::{Task, TaskId, TaskPriority, TaskStatus, ResourceAllocation};
use hardware_profiler::HardwareCapabilities;
use config::ConfigManager;

/// Main resource manager that integrates all resource management components
pub struct ResourceManager {
    /// Resource monitor
    monitor: Arc<ResourceMonitor>,
    
    /// Resource allocator
    allocator: Arc<ResourceAllocator>,
    
    /// Resource scheduler
    scheduler: Arc<ResourceScheduler>,
    
    /// Resource throttler
    throttler: Arc<ResourceThrottler>,
    
    /// Hardware capabilities
    hardware_capabilities: Arc<HardwareCapabilities>,
    
    /// Configuration manager
    config_manager: Arc<ConfigManager>,
}

impl ResourceManager {
    /// Creates a new resource manager
    pub fn new(
        hardware_capabilities: Arc<HardwareCapabilities>,
        config_manager: Arc<ConfigManager>,
    ) -> Result<Self> {
        // Create resource monitor
        let monitor = Arc::new(ResourceMonitor::new(
            hardware_capabilities.clone(),
            config_manager.clone(),
        )?);
        
        // Create resource allocator
        let allocator = Arc::new(ResourceAllocator::new(
            monitor.clone(),
            config_manager.clone(),
        )?);
        
        // Create resource scheduler
        let scheduler = Arc::new(ResourceScheduler::new(
            monitor.clone(),
            allocator.clone(),
            config_manager.clone(),
        )?);
        
        // Create resource throttler
        let throttler = Arc::new(ResourceThrottler::new(
            monitor.clone(),
            allocator.clone(),
            config_manager.clone(),
        )?);
        
        Ok(Self {
            monitor,
            allocator,
            scheduler,
            throttler,
            hardware_capabilities,
            config_manager,
        })
    }
    
    /// Starts the resource manager
    pub async fn start(&self) -> Result<()> {
        info!("Starting resource manager");
        
        // Start components
        self.monitor.start().await?;
        self.scheduler.start().await?;
        self.throttler.start().await?;
        
        info!("Resource manager started successfully");
        
        Ok(())
    }
    
    /// Stops the resource manager
    pub async fn stop(&self) -> Result<()> {
        info!("Stopping resource manager");
        
        // Stop components in reverse order
        self.throttler.stop().await?;
        self.scheduler.stop().await?;
        self.monitor.stop().await?;
        
        info!("Resource manager stopped successfully");
        
        Ok(())
    }
    
    /// Submits a task for execution
    pub async fn submit_task(&self, task: Task, priority: TaskPriority) -> Result<()> {
        self.scheduler.submit_task(task, priority).await
    }
    
    /// Cancels a task
    pub async fn cancel_task(&self, task_id: &TaskId) -> Result<()> {
        self.scheduler.cancel_task(task_id).await
    }
    
    /// Gets the status of a task
    pub fn get_task_status(&self, task_id: &TaskId) -> Result<TaskStatus> {
        self.scheduler.get_task_status(task_id)
    }
    
    /// Gets all tasks
    pub fn get_all_tasks(&self) -> Result<Vec<Task>> {
        self.scheduler.get_all_tasks()
    }
    
    /// Gets the current system resources
    pub async fn get_system_resources(&self) -> Result<common::models::SystemResources> {
        self.monitor.get_system_resources().await
    }
    
    /// Gets the current resource usage
    pub async fn get_resource_usage(&self) -> Result<ResourceUsage> {
        self.allocator.get_resource_usage().await
    }
    
    /// Gets the resource monitor
    pub fn get_monitor(&self) -> Arc<ResourceMonitor> {
        self.monitor.clone()
    }
    
    /// Gets the resource allocator
    pub fn get_allocator(&self) -> Arc<ResourceAllocator> {
        self.allocator.clone()
    }
    
    /// Gets the resource scheduler
    pub fn get_scheduler(&self) -> Arc<ResourceScheduler> {
        self.scheduler.clone()
    }
    
    /// Gets the resource throttler
    pub fn get_throttler(&self) -> Arc<ResourceThrottler> {
        self.throttler.clone()
    }
    
    /// Sets the throttling policy
    pub async fn set_throttling_policy(&self, policy: ThrottlingPolicy) -> Result<()> {
        self.throttler.set_policy(policy).await
    }
    
    /// Gets the throttling policy
    pub async fn get_throttling_policy(&self) -> ThrottlingPolicy {
        self.throttler.get_policy().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    // These tests would require mocking the dependencies
    // For now, we just have placeholder tests
    
    #[test]
    fn test_resource_manager() {
        // Placeholder test
        assert!(true);
    }
}

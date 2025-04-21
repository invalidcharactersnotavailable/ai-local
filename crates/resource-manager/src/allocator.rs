//! Resource allocation implementation
//!
//! This module provides functionality for allocating system resources
//! to tasks, with optimizations for different workloads.

use std::sync::Arc;
use tokio::sync::{RwLock, Mutex};
use anyhow::Result;
use tracing::{info, warn, error, debug, trace};
use std::collections::{HashMap, HashSet};
use dashmap::DashMap;
use uuid::Uuid;

use common::error::Error;
use common::models::{Task, TaskId, ResourceAllocation, SystemResources, GpuInfo};
use common::types::ResourceRequirements;
use config::ConfigManager;
use super::monitor::ResourceMonitor;

/// Resource allocator for managing system resources
pub struct ResourceAllocator {
    /// Resource monitor
    resource_monitor: Arc<ResourceMonitor>,
    
    /// Configuration manager
    config_manager: Arc<ConfigManager>,
    
    /// Allocated resources (task_id -> allocation)
    allocations: DashMap<TaskId, ResourceAllocation>,
    
    /// Allocation mutex to prevent race conditions
    allocation_mutex: Mutex<()>,
    
    /// Reserved CPU cores
    reserved_cpu_cores: HashSet<usize>,
    
    /// Reserved GPU devices
    reserved_gpu_devices: HashSet<usize>,
}

impl ResourceAllocator {
    /// Creates a new resource allocator
    pub fn new(
        resource_monitor: Arc<ResourceMonitor>,
        config_manager: Arc<ConfigManager>,
    ) -> Result<Self> {
        Ok(Self {
            resource_monitor,
            config_manager,
            allocations: DashMap::new(),
            allocation_mutex: Mutex::new(()),
            reserved_cpu_cores: HashSet::new(),
            reserved_gpu_devices: HashSet::new(),
        })
    }
    
    /// Allocates resources for a task
    pub async fn allocate_resources(&self, task: &Task) -> Result<ResourceAllocation> {
        // Acquire allocation mutex to prevent race conditions
        let _lock = self.allocation_mutex.lock().await;
        
        info!("Allocating resources for task {}", task.id);
        
        // Check if task already has an allocation
        if let Some(allocation) = self.allocations.get(&task.id) {
            return Ok(allocation.clone());
        }
        
        // Get task resource requirements
        let requirements = self.get_task_resource_requirements(task)?;
        
        // Get current system resources
        let resources = self.resource_monitor.get_system_resources().await?;
        
        // Check if we have enough resources
        if !self.resource_monitor.can_execute_task(task).await? {
            return Err(Error::Resource(format!(
                "Not enough resources to execute task {}",
                task.id
            )).into());
        }
        
        // Allocate CPU cores
        let cpu_cores = self.allocate_cpu_cores(requirements.min_cpu_cores)?;
        
        // Allocate GPU if required
        let gpu_devices = if requirements.requires_gpu {
            self.allocate_gpu_devices(requirements.min_gpu_memory_bytes, &resources.gpus)?
        } else {
            Vec::new()
        };
        
        // Create allocation
        let allocation = ResourceAllocation {
            task_id: task.id.clone(),
            cpu_cores: cpu_cores.clone(),
            memory_bytes: requirements.min_memory_bytes,
            gpu_devices: gpu_devices.clone(),
            gpu_memory_bytes: requirements.min_gpu_memory_bytes,
            disk_bytes: requirements.min_disk_bytes,
            allocation_time: chrono::Utc::now(),
        };
        
        // Update reserved resources
        for core in &cpu_cores {
            self.reserved_cpu_cores.insert(*core);
        }
        
        for device in &gpu_devices {
            self.reserved_gpu_devices.insert(*device);
        }
        
        // Store allocation
        self.allocations.insert(task.id.clone(), allocation.clone());
        
        info!(
            "Resources allocated for task {}: {} CPU cores, {} bytes memory, {} GPU devices",
            task.id,
            cpu_cores.len(),
            requirements.min_memory_bytes,
            gpu_devices.len()
        );
        
        Ok(allocation)
    }
    
    /// Releases resources for a task
    pub async fn release_resources(&self, task_id: &TaskId) -> Result<()> {
        // Acquire allocation mutex to prevent race conditions
        let _lock = self.allocation_mutex.lock().await;
        
        info!("Releasing resources for task {}", task_id);
        
        // Check if task has an allocation
        if let Some((_, allocation)) = self.allocations.remove(task_id) {
            // Release CPU cores
            for core in &allocation.cpu_cores {
                self.reserved_cpu_cores.remove(core);
            }
            
            // Release GPU devices
            for device in &allocation.gpu_devices {
                self.reserved_gpu_devices.remove(device);
            }
            
            info!(
                "Resources released for task {}: {} CPU cores, {} GPU devices",
                task_id,
                allocation.cpu_cores.len(),
                allocation.gpu_devices.len()
            );
            
            Ok(())
        } else {
            Err(Error::NotFound(format!("No allocation found for task {}", task_id)).into())
        }
    }
    
    /// Gets resource allocation for a task
    pub fn get_allocation(&self, task_id: &TaskId) -> Option<ResourceAllocation> {
        self.allocations.get(task_id).map(|a| a.clone())
    }
    
    /// Gets all resource allocations
    pub fn get_all_allocations(&self) -> Vec<ResourceAllocation> {
        self.allocations.iter().map(|entry| entry.value().clone()).collect()
    }
    
    /// Gets task resource requirements
    fn get_task_resource_requirements(&self, task: &Task) -> Result<ResourceRequirements> {
        // In a real implementation, this would be more sophisticated
        // For now, we just use some default values based on task type
        
        match task.task_type {
            common::models::TaskType::Inference => {
                Ok(ResourceRequirements {
                    min_cpu_cores: 1,
                    min_memory_bytes: 1024 * 1024 * 1024, // 1 GB
                    min_gpu_memory_bytes: Some(1024 * 1024 * 1024), // 1 GB
                    min_disk_bytes: 100 * 1024 * 1024, // 100 MB
                    requires_gpu: true,
                    required_cpu_features: vec!["sse4.1".to_string(), "avx".to_string()],
                    required_gpu_features: vec![],
                })
            },
            common::models::TaskType::FineTuning => {
                Ok(ResourceRequirements {
                    min_cpu_cores: 4,
                    min_memory_bytes: 8 * 1024 * 1024 * 1024, // 8 GB
                    min_gpu_memory_bytes: Some(8 * 1024 * 1024 * 1024), // 8 GB
                    min_disk_bytes: 10 * 1024 * 1024 * 1024, // 10 GB
                    requires_gpu: true,
                    required_cpu_features: vec!["sse4.1".to_string(), "avx".to_string(), "avx2".to_string()],
                    required_gpu_features: vec![],
                })
            },
            common::models::TaskType::Evaluation => {
                Ok(ResourceRequirements {
                    min_cpu_cores: 2,
                    min_memory_bytes: 4 * 1024 * 1024 * 1024, // 4 GB
                    min_gpu_memory_bytes: Some(4 * 1024 * 1024 * 1024), // 4 GB
                    min_disk_bytes: 1 * 1024 * 1024 * 1024, // 1 GB
                    requires_gpu: true,
                    required_cpu_features: vec!["sse4.1".to_string(), "avx".to_string()],
                    required_gpu_features: vec![],
                })
            },
            common::models::TaskType::Export => {
                Ok(ResourceRequirements {
                    min_cpu_cores: 2,
                    min_memory_bytes: 4 * 1024 * 1024 * 1024, // 4 GB
                    min_gpu_memory_bytes: None,
                    min_disk_bytes: 10 * 1024 * 1024 * 1024, // 10 GB
                    requires_gpu: false,
                    required_cpu_features: vec![],
                    required_gpu_features: vec![],
                })
            },
            common::models::TaskType::Custom(_) => {
                // For custom tasks, we use conservative defaults
                Ok(ResourceRequirements {
                    min_cpu_cores: 2,
                    min_memory_bytes: 2 * 1024 * 1024 * 1024, // 2 GB
                    min_gpu_memory_bytes: None,
                    min_disk_bytes: 1 * 1024 * 1024 * 1024, // 1 GB
                    requires_gpu: false,
                    required_cpu_features: vec![],
                    required_gpu_features: vec![],
                })
            },
        }
    }
    
    /// Allocates CPU cores
    fn allocate_cpu_cores(&self, count: usize) -> Result<Vec<usize>> {
        let total_cores = self.resource_monitor.get_total_cpu_cores()?;
        let mut allocated = Vec::new();
        
        // Find available cores
        for core in 0..total_cores {
            if !self.reserved_cpu_cores.contains(&core) {
                allocated.push(core);
                
                if allocated.len() >= count {
                    break;
                }
            }
        }
        
        if allocated.len() < count {
            return Err(Error::Resource(format!(
                "Not enough CPU cores available: required {}, available {}",
                count, allocated.len()
            )).into());
        }
        
        Ok(allocated)
    }
    
    /// Allocates GPU devices
    fn allocate_gpu_devices(&self, min_memory: Option<u64>, gpus: &[GpuInfo]) -> Result<Vec<usize>> {
        if gpus.is_empty() {
            return Err(Error::Resource("No GPUs available".to_string()).into());
        }
        
        // If no minimum memory specified, allocate any available GPU
        if min_memory.is_none() {
            for gpu in gpus {
                if !self.reserved_gpu_devices.contains(&gpu.index) {
                    return Ok(vec![gpu.index]);
                }
            }
            
            return Err(Error::Resource("All GPUs are reserved".to_string()).into());
        }
        
        let required_memory = min_memory.unwrap();
        
        // Find GPU with enough memory
        for gpu in gpus {
            if !self.reserved_gpu_devices.contains(&gpu.index) && gpu.available_memory_bytes >= required_memory {
                return Ok(vec![gpu.index]);
            }
        }
        
        // If no single GPU has enough memory, try to find multiple GPUs
        // In a real implementation, we would check if the task supports multi-GPU
        // For now, we just return an error
        
        Err(Error::Resource(format!(
            "No GPU with enough memory available: required {} bytes",
            required_memory
        )).into())
    }
    
    /// Checks if a task can be executed with current resources
    pub async fn can_execute_task(&self, task: &Task) -> Result<bool> {
        self.resource_monitor.can_execute_task(task).await
    }
    
    /// Gets the current resource usage
    pub async fn get_resource_usage(&self) -> Result<ResourceUsage> {
        // Get current system resources
        let resources = self.resource_monitor.get_system_resources().await?;
        
        // Calculate usage
        let cpu_usage = (self.reserved_cpu_cores.len() as f32) / (resources.cpu.logical_cores as f32);
        let memory_usage = self.calculate_allocated_memory() as f32 / resources.memory.total_bytes as f32;
        let gpu_usage = if resources.gpus.is_empty() {
            0.0
        } else {
            (self.reserved_gpu_devices.len() as f32) / (resources.gpus.len() as f32)
        };
        
        Ok(ResourceUsage {
            cpu_usage,
            memory_usage,
            gpu_usage,
            allocated_cpu_cores: self.reserved_cpu_cores.len(),
            allocated_memory_bytes: self.calculate_allocated_memory(),
            allocated_gpu_devices: self.reserved_gpu_devices.len(),
            total_allocations: self.allocations.len(),
        })
    }
    
    /// Calculates the total allocated memory
    fn calculate_allocated_memory(&self) -> u64 {
        self.allocations.iter().map(|entry| entry.value().memory_bytes).sum()
    }
}

/// Resource usage information
#[derive(Debug, Clone)]
pub struct ResourceUsage {
    /// CPU usage (0.0-1.0)
    pub cpu_usage: f32,
    
    /// Memory usage (0.0-1.0)
    pub memory_usage: f32,
    
    /// GPU usage (0.0-1.0)
    pub gpu_usage: f32,
    
    /// Number of allocated CPU cores
    pub allocated_cpu_cores: usize,
    
    /// Allocated memory in bytes
    pub allocated_memory_bytes: u64,
    
    /// Number of allocated GPU devices
    pub allocated_gpu_devices: usize,
    
    /// Total number of allocations
    pub total_allocations: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    // These tests would require mocking the dependencies
    // For now, we just have placeholder tests
    
    #[test]
    fn test_resource_allocator() {
        // Placeholder test
        assert!(true);
    }
}

//! Resource throttling implementation
//!
//! This module provides functionality for throttling resource usage
//! to prevent system overload and ensure fair resource allocation.

use std::sync::Arc;
use tokio::sync::{RwLock, Mutex};
use anyhow::Result;
use tracing::{info, warn, error, debug, trace};
use std::collections::HashMap;
use dashmap::DashMap;
use std::time::{Duration, Instant};

use common::error::Error;
use common::models::{Task, TaskId, ResourceAllocation};
use config::ConfigManager;
use super::monitor::ResourceMonitor;
use super::allocator::ResourceAllocator;

/// Throttling policy for resource usage
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThrottlingPolicy {
    /// No throttling
    None,
    
    /// Gradual throttling based on resource usage
    Gradual,
    
    /// Aggressive throttling when thresholds are exceeded
    Aggressive,
}

/// Resource throttler for managing system load
pub struct ResourceThrottler {
    /// Resource monitor
    resource_monitor: Arc<ResourceMonitor>,
    
    /// Resource allocator
    resource_allocator: Arc<ResourceAllocator>,
    
    /// Configuration manager
    config_manager: Arc<ConfigManager>,
    
    /// Throttling policy
    policy: RwLock<ThrottlingPolicy>,
    
    /// CPU usage threshold for throttling (percentage)
    cpu_threshold: f32,
    
    /// Memory usage threshold for throttling (percentage)
    memory_threshold: f32,
    
    /// GPU usage threshold for throttling (percentage)
    gpu_threshold: f32,
    
    /// Task throttling state (task_id -> throttling factor)
    throttled_tasks: DashMap<TaskId, f32>,
    
    /// Throttling check interval
    check_interval: Duration,
    
    /// Throttler running flag
    running: Arc<RwLock<bool>>,
}

impl ResourceThrottler {
    /// Creates a new resource throttler
    pub fn new(
        resource_monitor: Arc<ResourceMonitor>,
        resource_allocator: Arc<ResourceAllocator>,
        config_manager: Arc<ConfigManager>,
    ) -> Result<Self> {
        // Get throttling policy from config
        let policy_str = config_manager
            .get_string("throttling_policy")
            .unwrap_or_else(|_| "gradual".to_string());
        
        let policy = match policy_str.to_lowercase().as_str() {
            "none" => ThrottlingPolicy::None,
            "aggressive" => ThrottlingPolicy::Aggressive,
            _ => ThrottlingPolicy::Gradual,
        };
        
        // Get thresholds from config
        let cpu_threshold = config_manager
            .get_f32("cpu_throttling_threshold")
            .unwrap_or(80.0);
        
        let memory_threshold = config_manager
            .get_f32("memory_throttling_threshold")
            .unwrap_or(80.0);
        
        let gpu_threshold = config_manager
            .get_f32("gpu_throttling_threshold")
            .unwrap_or(80.0);
        
        // Get check interval from config
        let check_interval = config_manager
            .get_duration("throttling_check_interval")
            .unwrap_or_else(|_| Duration::from_secs(1));
        
        Ok(Self {
            resource_monitor,
            resource_allocator,
            config_manager,
            policy: RwLock::new(policy),
            cpu_threshold,
            memory_threshold,
            gpu_threshold,
            throttled_tasks: DashMap::new(),
            check_interval,
            running: Arc::new(RwLock::new(false)),
        })
    }
    
    /// Starts the throttler
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
        
        info!("Starting resource throttler with policy {:?}", *self.policy.read().await);
        
        // Start throttling loop
        self.start_throttling_loop().await?;
        
        info!("Resource throttler started successfully");
        
        Ok(())
    }
    
    /// Stops the throttler
    pub async fn stop(&self) -> Result<()> {
        // Check if running
        {
            let running = self.running.read().await;
            if !*running {
                return Ok(());
            }
        }
        
        info!("Stopping resource throttler");
        
        // Set running flag to false
        {
            let mut running = self.running.write().await;
            *running = false;
        }
        
        // Clear throttled tasks
        self.throttled_tasks.clear();
        
        info!("Resource throttler stopped successfully");
        
        Ok(())
    }
    
    /// Starts the throttling loop
    async fn start_throttling_loop(&self) -> Result<()> {
        let resource_monitor = self.resource_monitor.clone();
        let resource_allocator = self.resource_allocator.clone();
        let policy = self.policy.clone();
        let cpu_threshold = self.cpu_threshold;
        let memory_threshold = self.memory_threshold;
        let gpu_threshold = self.gpu_threshold;
        let throttled_tasks = self.throttled_tasks.clone();
        let check_interval = self.check_interval;
        let running = self.running.clone();
        
        tokio::spawn(async move {
            while {
                let is_running = *running.read().await;
                is_running
            } {
                // Get current policy
                let current_policy = *policy.read().await;
                
                // Skip if policy is None
                if current_policy == ThrottlingPolicy::None {
                    tokio::time::sleep(check_interval).await;
                    continue;
                }
                
                // Get current system resources
                match resource_monitor.get_system_resources().await {
                    Ok(resources) => {
                        // Check if we need to throttle
                        let cpu_usage = resources.cpu.usage_percent;
                        let memory_usage = resources.memory.usage_percent;
                        let gpu_usage = resources.gpus.iter().map(|g| g.usage_percent).max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)).unwrap_or(0.0);
                        
                        let need_throttling = cpu_usage > cpu_threshold || 
                                             memory_usage > memory_threshold || 
                                             gpu_usage > gpu_threshold;
                        
                        if need_throttling {
                            // Calculate throttling factor
                            let cpu_factor = if cpu_usage > cpu_threshold {
                                (cpu_usage - cpu_threshold) / (100.0 - cpu_threshold)
                            } else {
                                0.0
                            };
                            
                            let memory_factor = if memory_usage > memory_threshold {
                                (memory_usage - memory_threshold) / (100.0 - memory_threshold)
                            } else {
                                0.0
                            };
                            
                            let gpu_factor = if gpu_usage > gpu_threshold {
                                (gpu_usage - gpu_threshold) / (100.0 - gpu_threshold)
                            } else {
                                0.0
                            };
                            
                            // Use maximum factor
                            let throttling_factor = cpu_factor.max(memory_factor).max(gpu_factor);
                            
                            // Apply throttling based on policy
                            match current_policy {
                                ThrottlingPolicy::Gradual => {
                                    // Apply gradual throttling to all running tasks
                                    let allocations = resource_allocator.get_all_allocations();
                                    
                                    for allocation in allocations {
                                        throttled_tasks.insert(allocation.task_id.clone(), throttling_factor);
                                    }
                                    
                                    info!(
                                        "Applied gradual throttling (factor: {:.2}) due to resource usage: CPU {:.1}%, Memory {:.1}%, GPU {:.1}%",
                                        throttling_factor, cpu_usage, memory_usage, gpu_usage
                                    );
                                },
                                ThrottlingPolicy::Aggressive => {
                                    // Apply aggressive throttling to all running tasks
                                    let allocations = resource_allocator.get_all_allocations();
                                    
                                    for allocation in allocations {
                                        throttled_tasks.insert(allocation.task_id.clone(), 0.8); // 80% throttling
                                    }
                                    
                                    warn!(
                                        "Applied aggressive throttling due to resource usage: CPU {:.1}%, Memory {:.1}%, GPU {:.1}%",
                                        cpu_usage, memory_usage, gpu_usage
                                    );
                                },
                                ThrottlingPolicy::None => {
                                    // Should not happen, but just in case
                                    throttled_tasks.clear();
                                }
                            }
                        } else {
                            // No need for throttling, clear throttled tasks
                            if !throttled_tasks.is_empty() {
                                throttled_tasks.clear();
                                debug!("Removed throttling as resource usage is below thresholds");
                            }
                        }
                    },
                    Err(e) => {
                        warn!("Failed to get system resources for throttling check: {}", e);
                    }
                }
                
                // Sleep for check interval
                tokio::time::sleep(check_interval).await;
            }
        });
        
        Ok(())
    }
    
    /// Gets the throttling factor for a task
    pub fn get_throttling_factor(&self, task_id: &TaskId) -> f32 {
        match self.throttled_tasks.get(task_id) {
            Some(factor) => *factor,
            None => 0.0, // No throttling
        }
    }
    
    /// Checks if a task is throttled
    pub fn is_task_throttled(&self, task_id: &TaskId) -> bool {
        self.throttled_tasks.contains_key(task_id)
    }
    
    /// Gets the current throttling policy
    pub async fn get_policy(&self) -> ThrottlingPolicy {
        *self.policy.read().await
    }
    
    /// Sets the throttling policy
    pub async fn set_policy(&self, policy: ThrottlingPolicy) -> Result<()> {
        // Update policy
        {
            let mut current_policy = self.policy.write().await;
            *current_policy = policy;
        }
        
        // Clear throttled tasks if policy is None
        if policy == ThrottlingPolicy::None {
            self.throttled_tasks.clear();
        }
        
        info!("Throttling policy set to {:?}", policy);
        
        Ok(())
    }
    
    /// Gets the number of throttled tasks
    pub fn get_throttled_tasks_count(&self) -> usize {
        self.throttled_tasks.len()
    }
    
    /// Gets all throttled tasks
    pub fn get_all_throttled_tasks(&self) -> HashMap<TaskId, f32> {
        self.throttled_tasks.iter().map(|entry| (entry.key().clone(), *entry.value())).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    // These tests would require mocking the dependencies
    // For now, we just have placeholder tests
    
    #[test]
    fn test_resource_throttler() {
        // Placeholder test
        assert!(true);
    }
}

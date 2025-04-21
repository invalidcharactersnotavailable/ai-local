//! Resource scheduler implementation
//!
//! This module provides functionality for scheduling tasks based on
//! available resources and priorities.

use std::sync::Arc;
use tokio::sync::{RwLock, Mutex, mpsc};
use anyhow::Result;
use tracing::{info, warn, error, debug, trace};
use std::collections::{HashMap, VecDeque, BinaryHeap};
use std::cmp::{Ordering, Reverse};
use dashmap::DashMap;
use uuid::Uuid;
use std::time::{Duration, Instant};

use common::error::Error;
use common::models::{Task, TaskId, TaskPriority, TaskStatus, ResourceAllocation};
use common::types::ResourceRequirements;
use config::ConfigManager;
use super::monitor::ResourceMonitor;
use super::allocator::ResourceAllocator;

/// Task with scheduling information
#[derive(Debug, Clone)]
struct ScheduledTask {
    /// Task information
    task: Task,
    
    /// Scheduling priority
    priority: TaskPriority,
    
    /// Submission time
    submission_time: chrono::DateTime<chrono::Utc>,
    
    /// Estimated execution time (seconds)
    estimated_execution_time: u64,
    
    /// Actual start time (if started)
    start_time: Option<chrono::DateTime<chrono::Utc>>,
    
    /// Resource allocation (if allocated)
    allocation: Option<ResourceAllocation>,
}

impl PartialEq for ScheduledTask {
    fn eq(&self, other: &Self) -> bool {
        self.task.id == other.task.id
    }
}

impl Eq for ScheduledTask {}

impl PartialOrd for ScheduledTask {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ScheduledTask {
    fn cmp(&self, other: &Self) -> Ordering {
        // First compare by priority (higher priority first)
        match self.priority.cmp(&other.priority) {
            Ordering::Equal => {},
            ordering => return ordering.reverse(), // Reverse because BinaryHeap is a max-heap
        }
        
        // Then compare by submission time (earlier first)
        match self.submission_time.cmp(&other.submission_time) {
            Ordering::Equal => {},
            ordering => return ordering,
        }
        
        // Finally compare by ID for stable ordering
        self.task.id.cmp(&other.task.id)
    }
}

/// Resource scheduler for scheduling tasks based on available resources
pub struct ResourceScheduler {
    /// Resource monitor
    resource_monitor: Arc<ResourceMonitor>,
    
    /// Resource allocator
    resource_allocator: Arc<ResourceAllocator>,
    
    /// Configuration manager
    config_manager: Arc<ConfigManager>,
    
    /// Task queue
    task_queue: Arc<Mutex<BinaryHeap<Reverse<ScheduledTask>>>>,
    
    /// Running tasks
    running_tasks: DashMap<TaskId, ScheduledTask>,
    
    /// Completed tasks
    completed_tasks: DashMap<TaskId, ScheduledTask>,
    
    /// Task status updates sender
    status_tx: mpsc::Sender<(TaskId, TaskStatus)>,
    
    /// Task status updates receiver
    status_rx: Arc<Mutex<mpsc::Receiver<(TaskId, TaskStatus)>>>,
    
    /// Maximum concurrent tasks
    max_concurrent_tasks: usize,
    
    /// Scheduler running flag
    running: Arc<RwLock<bool>>,
    
    /// Scheduler interval
    scheduler_interval: Duration,
}

impl ResourceScheduler {
    /// Creates a new resource scheduler
    pub fn new(
        resource_monitor: Arc<ResourceMonitor>,
        resource_allocator: Arc<ResourceAllocator>,
        config_manager: Arc<ConfigManager>,
    ) -> Result<Self> {
        // Get maximum concurrent tasks from config
        let max_concurrent_tasks = config_manager
            .get_usize("max_concurrent_tasks")
            .unwrap_or_else(|_| {
                // Default to number of CPU cores
                match resource_monitor.get_total_cpu_cores() {
                    Ok(cores) => cores,
                    Err(_) => 4, // Fallback default
                }
            });
        
        // Get scheduler interval from config
        let scheduler_interval = config_manager
            .get_duration("scheduler_interval")
            .unwrap_or_else(|_| Duration::from_millis(100));
        
        // Create channel for task status updates
        let (status_tx, status_rx) = mpsc::channel(100);
        
        Ok(Self {
            resource_monitor,
            resource_allocator,
            config_manager,
            task_queue: Arc::new(Mutex::new(BinaryHeap::new())),
            running_tasks: DashMap::new(),
            completed_tasks: DashMap::new(),
            status_tx,
            status_rx: Arc::new(Mutex::new(status_rx)),
            max_concurrent_tasks,
            running: Arc::new(RwLock::new(false)),
            scheduler_interval,
        })
    }
    
    /// Starts the scheduler
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
        
        info!("Starting resource scheduler");
        
        // Start scheduler loop
        self.start_scheduler_loop().await?;
        
        // Start status update handler
        self.start_status_update_handler().await?;
        
        info!("Resource scheduler started successfully");
        
        Ok(())
    }
    
    /// Stops the scheduler
    pub async fn stop(&self) -> Result<()> {
        // Check if running
        {
            let running = self.running.read().await;
            if !*running {
                return Ok(());
            }
        }
        
        info!("Stopping resource scheduler");
        
        // Set running flag to false
        {
            let mut running = self.running.write().await;
            *running = false;
        }
        
        info!("Resource scheduler stopped successfully");
        
        Ok(())
    }
    
    /// Starts the scheduler loop
    async fn start_scheduler_loop(&self) -> Result<()> {
        let task_queue = self.task_queue.clone();
        let running_tasks = self.running_tasks.clone();
        let resource_allocator = self.resource_allocator.clone();
        let resource_monitor = self.resource_monitor.clone();
        let status_tx = self.status_tx.clone();
        let running = self.running.clone();
        let max_concurrent_tasks = self.max_concurrent_tasks;
        let scheduler_interval = self.scheduler_interval;
        
        tokio::spawn(async move {
            while {
                let is_running = *running.read().await;
                is_running
            } {
                // Check if we can start more tasks
                let current_running = running_tasks.len();
                
                if current_running < max_concurrent_tasks {
                    // Get next task from queue
                    let next_task = {
                        let mut queue = task_queue.lock().await;
                        queue.pop().map(|t| t.0)
                    };
                    
                    if let Some(scheduled_task) = next_task {
                        let task = &scheduled_task.task;
                        
                        // Check if we have enough resources
                        match resource_monitor.can_execute_task(task).await {
                            Ok(true) => {
                                // Allocate resources
                                match resource_allocator.allocate_resources(task).await {
                                    Ok(allocation) => {
                                        // Update task with allocation and start time
                                        let mut updated_task = scheduled_task.clone();
                                        updated_task.allocation = Some(allocation);
                                        updated_task.start_time = Some(chrono::Utc::now());
                                        
                                        // Add to running tasks
                                        running_tasks.insert(task.id.clone(), updated_task);
                                        
                                        // Send status update
                                        let _ = status_tx.send((task.id.clone(), TaskStatus::Running)).await;
                                        
                                        info!("Started task {} (priority: {:?})", task.id, scheduled_task.priority);
                                    },
                                    Err(e) => {
                                        // Failed to allocate resources, put back in queue
                                        warn!("Failed to allocate resources for task {}: {}", task.id, e);
                                        
                                        let mut queue = task_queue.lock().await;
                                        queue.push(Reverse(scheduled_task));
                                        
                                        // Send status update
                                        let _ = status_tx.send((task.id.clone(), TaskStatus::Queued)).await;
                                    }
                                }
                            },
                            Ok(false) => {
                                // Not enough resources, put back in queue
                                debug!("Not enough resources for task {}, keeping in queue", task.id);
                                
                                let mut queue = task_queue.lock().await;
                                queue.push(Reverse(scheduled_task));
                            },
                            Err(e) => {
                                // Error checking resources, put back in queue
                                warn!("Error checking resources for task {}: {}", task.id, e);
                                
                                let mut queue = task_queue.lock().await;
                                queue.push(Reverse(scheduled_task));
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
    
    /// Starts the status update handler
    async fn start_status_update_handler(&self) -> Result<()> {
        let status_rx = self.status_rx.clone();
        let running_tasks = self.running_tasks.clone();
        let completed_tasks = self.completed_tasks.clone();
        let resource_allocator = self.resource_allocator.clone();
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
                            // Task is done, release resources
                            if let Some((_, task)) = running_tasks.remove(&task_id) {
                                // Release resources
                                let _ = resource_allocator.release_resources(&task_id).await;
                                
                                // Add to completed tasks
                                completed_tasks.insert(task_id.clone(), task);
                                
                                info!("Task {} completed with status {:?}", task_id, status);
                            }
                        },
                        _ => {
                            // Other status updates don't require resource changes
                        }
                    }
                }
            }
        });
        
        Ok(())
    }
    
    /// Submits a task for scheduling
    pub async fn submit_task(&self, task: Task, priority: TaskPriority) -> Result<()> {
        // Check if task already exists
        if self.running_tasks.contains_key(&task.id) || self.completed_tasks.contains_key(&task.id) {
            return Err(Error::AlreadyExists(format!("Task {} already exists", task.id)).into());
        }
        
        // Create scheduled task
        let scheduled_task = ScheduledTask {
            task: task.clone(),
            priority,
            submission_time: chrono::Utc::now(),
            estimated_execution_time: self.estimate_execution_time(&task)?,
            start_time: None,
            allocation: None,
        };
        
        // Add to queue
        {
            let mut queue = self.task_queue.lock().await;
            queue.push(Reverse(scheduled_task));
        }
        
        // Send status update
        let _ = self.status_tx.send((task.id.clone(), TaskStatus::Queued)).await;
        
        info!("Submitted task {} with priority {:?}", task.id, priority);
        
        Ok(())
    }
    
    /// Cancels a task
    pub async fn cancel_task(&self, task_id: &TaskId) -> Result<()> {
        // Check if task is in queue
        {
            let mut queue = self.task_queue.lock().await;
            let mut new_queue = BinaryHeap::new();
            
            // Rebuild queue without the cancelled task
            while let Some(Reverse(task)) = queue.pop() {
                if task.task.id != *task_id {
                    new_queue.push(Reverse(task));
                } else {
                    // Found the task to cancel
                    info!("Cancelled queued task {}", task_id);
                    
                    // Send status update
                    let _ = self.status_tx.send((task_id.clone(), TaskStatus::Cancelled)).await;
                    
                    // Add to completed tasks
                    self.completed_tasks.insert(task_id.clone(), task);
                    
                    *queue = new_queue;
                    return Ok(());
                }
            }
            
            *queue = new_queue;
        }
        
        // Check if task is running
        if self.running_tasks.contains_key(task_id) {
            // Send status update
            let _ = self.status_tx.send((task_id.clone(), TaskStatus::Cancelled)).await;
            
            info!("Cancelled running task {}", task_id);
            
            return Ok(());
        }
        
        // Check if task is completed
        if self.completed_tasks.contains_key(task_id) {
            return Err(Error::InvalidArgument(format!("Task {} is already completed", task_id)).into());
        }
        
        Err(Error::NotFound(format!("Task {} not found", task_id)).into())
    }
    
    /// Gets the status of a task
    pub fn get_task_status(&self, task_id: &TaskId) -> Result<TaskStatus> {
        // Check if task is running
        if self.running_tasks.contains_key(task_id) {
            return Ok(TaskStatus::Running);
        }
        
        // Check if task is completed
        if let Some(task) = self.completed_tasks.get(task_id) {
            return Ok(task.task.status);
        }
        
        // Check if task is in queue
        {
            let queue = match self.task_queue.try_lock() {
                Ok(queue) => queue,
                Err(_) => return Err(Error::Internal("Failed to acquire lock on task queue".to_string()).into()),
            };
            
            for Reverse(task) in queue.iter() {
                if task.task.id == *task_id {
                    return Ok(TaskStatus::Queued);
                }
            }
        }
        
        Err(Error::NotFound(format!("Task {} not found", task_id)).into())
    }
    
    /// Gets all tasks
    pub fn get_all_tasks(&self) -> Result<Vec<Task>> {
        let mut tasks = Vec::new();
        
        // Get queued tasks
        {
            let queue = match self.task_queue.try_lock() {
                Ok(queue) => queue,
                Err(_) => return Err(Error::Internal("Failed to acquire lock on task queue".to_string()).into()),
            };
            
            for Reverse(task) in queue.iter() {
                tasks.push(task.task.clone());
            }
        }
        
        // Get running tasks
        for entry in self.running_tasks.iter() {
            tasks.push(entry.task.clone());
        }
        
        // Get completed tasks
        for entry in self.completed_tasks.iter() {
            tasks.push(entry.task.clone());
        }
        
        Ok(tasks)
    }
    
    /// Gets the number of queued tasks
    pub fn get_queue_length(&self) -> Result<usize> {
        let queue = match self.task_queue.try_lock() {
            Ok(queue) => queue,
            Err(_) => return Err(Error::Internal("Failed to acquire lock on task queue".to_string()).into()),
        };
        
        Ok(queue.len())
    }
    
    /// Gets the number of running tasks
    pub fn get_running_tasks_count(&self) -> usize {
        self.running_tasks.len()
    }
    
    /// Gets the number of completed tasks
    pub fn get_completed_tasks_count(&self) -> usize {
        self.completed_tasks.len()
    }
    
    /// Estimates execution time for a task
    fn estimate_execution_time(&self, task: &Task) -> Result<u64> {
        // In a real implementation, this would be more sophisticated
        // For now, we just use some default values based on task type
        
        match task.task_type {
            common::models::TaskType::Inference => Ok(5),
            common::models::TaskType::FineTuning => Ok(3600),
            common::models::TaskType::Evaluation => Ok(300),
            common::models::TaskType::Export => Ok(60),
            common::models::TaskType::Custom(_) => Ok(60),
        }
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
    fn test_resource_scheduler() {
        // Placeholder test
        assert!(true);
    }
}

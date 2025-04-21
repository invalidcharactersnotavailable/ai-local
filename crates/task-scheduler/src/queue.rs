//! Task queue management
//!
//! This module provides functionality for managing task queues,
//! including prioritization, dependency resolution, and scheduling.

use std::sync::Arc;
use tokio::sync::{RwLock, Mutex};
use anyhow::Result;
use tracing::{info, warn, error, debug, trace};
use std::collections::{HashMap, HashSet, BinaryHeap, VecDeque};
use std::cmp::{Ordering, Reverse};
use dashmap::DashMap;
use uuid::Uuid;
use chrono::{DateTime, Utc};

use common::error::Error;
use common::models::{Task, TaskId, TaskType, TaskStatus, TaskPriority};
use config::ConfigManager;

/// Task with queue information
#[derive(Debug, Clone)]
struct QueuedTask {
    /// Task information
    task: Task,
    
    /// Queue priority
    priority: TaskPriority,
    
    /// Submission time
    submission_time: DateTime<Utc>,
    
    /// Dependencies that are not yet satisfied
    pending_dependencies: HashSet<TaskId>,
}

impl PartialEq for QueuedTask {
    fn eq(&self, other: &Self) -> bool {
        self.task.id == other.task.id
    }
}

impl Eq for QueuedTask {}

impl PartialOrd for QueuedTask {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for QueuedTask {
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

/// Task queue manager for handling task queues
pub struct TaskQueueManager {
    /// Ready queue (tasks that can be executed)
    ready_queue: Arc<Mutex<BinaryHeap<Reverse<QueuedTask>>>>,
    
    /// Waiting queue (tasks with pending dependencies)
    waiting_queue: DashMap<TaskId, QueuedTask>,
    
    /// Dependency graph (task_id -> tasks that depend on it)
    dependency_graph: DashMap<TaskId, HashSet<TaskId>>,
    
    /// Configuration manager
    config_manager: Arc<ConfigManager>,
    
    /// Maximum queue size
    max_queue_size: usize,
}

impl TaskQueueManager {
    /// Creates a new task queue manager
    pub fn new(config_manager: Arc<ConfigManager>) -> Result<Self> {
        // Get maximum queue size from config
        let max_queue_size = config_manager
            .get_usize("max_task_queue_size")
            .unwrap_or(1000);
        
        Ok(Self {
            ready_queue: Arc::new(Mutex::new(BinaryHeap::new())),
            waiting_queue: DashMap::new(),
            dependency_graph: DashMap::new(),
            config_manager,
            max_queue_size,
        })
    }
    
    /// Enqueues a task
    pub async fn enqueue_task(&self, task: Task, priority: TaskPriority) -> Result<()> {
        // Check if queue is full
        if self.get_queue_size() >= self.max_queue_size {
            return Err(Error::Resource("Task queue is full".to_string()).into());
        }
        
        // Check if task already exists
        if self.is_task_queued(&task.id) {
            return Err(Error::AlreadyExists(format!("Task {} is already queued", task.id)).into());
        }
        
        // Create queued task
        let mut queued_task = QueuedTask {
            task: task.clone(),
            priority,
            submission_time: Utc::now(),
            pending_dependencies: HashSet::new(),
        };
        
        // Check dependencies
        let has_pending_dependencies = !task.dependencies.is_empty();
        
        if has_pending_dependencies {
            // Add pending dependencies
            for dep_id in &task.dependencies {
                // Add to dependency graph
                self.dependency_graph
                    .entry(dep_id.clone())
                    .or_insert_with(HashSet::new)
                    .insert(task.id.clone());
                
                // Check if dependency is completed
                if !self.is_dependency_satisfied(dep_id) {
                    queued_task.pending_dependencies.insert(dep_id.clone());
                }
            }
            
            // If there are pending dependencies, add to waiting queue
            if !queued_task.pending_dependencies.is_empty() {
                self.waiting_queue.insert(task.id.clone(), queued_task);
                
                debug!(
                    "Task {} added to waiting queue with {} pending dependencies",
                    task.id,
                    queued_task.pending_dependencies.len()
                );
                
                return Ok(());
            }
        }
        
        // No pending dependencies, add to ready queue
        {
            let mut queue = self.ready_queue.lock().await;
            queue.push(Reverse(queued_task));
        }
        
        debug!("Task {} added to ready queue with priority {:?}", task.id, priority);
        
        Ok(())
    }
    
    /// Dequeues the next task
    pub async fn dequeue_task(&self) -> Option<Task> {
        let mut queue = self.ready_queue.lock().await;
        
        queue.pop().map(|Reverse(queued_task)| queued_task.task)
    }
    
    /// Peeks at the next task without removing it
    pub async fn peek_task(&self) -> Option<Task> {
        let queue = self.ready_queue.lock().await;
        
        queue.peek().map(|Reverse(queued_task)| queued_task.task.clone())
    }
    
    /// Notifies that a task has completed
    pub async fn notify_task_completed(&self, task_id: &TaskId) -> Result<()> {
        // Get dependent tasks
        let dependent_tasks = self.get_dependent_tasks(task_id);
        
        // Update waiting tasks
        for dep_task_id in dependent_tasks {
            if let Some(mut entry) = self.waiting_queue.get_mut(&dep_task_id) {
                // Remove this dependency
                entry.pending_dependencies.remove(task_id);
                
                // Check if all dependencies are satisfied
                if entry.pending_dependencies.is_empty() {
                    // Move to ready queue
                    let queued_task = entry.clone();
                    
                    {
                        let mut queue = self.ready_queue.lock().await;
                        queue.push(Reverse(queued_task));
                    }
                    
                    // Remove from waiting queue
                    self.waiting_queue.remove(&dep_task_id);
                    
                    debug!("Task {} moved from waiting to ready queue", dep_task_id);
                }
            }
        }
        
        // Remove from dependency graph
        self.dependency_graph.remove(task_id);
        
        Ok(())
    }
    
    /// Cancels a task
    pub async fn cancel_task(&self, task_id: &TaskId) -> Result<Option<Task>> {
        // Check if task is in waiting queue
        if let Some((_, queued_task)) = self.waiting_queue.remove(task_id) {
            // Get dependent tasks
            let dependent_tasks = self.get_dependent_tasks(task_id);
            
            // Mark dependent tasks as failed
            for dep_task_id in dependent_tasks {
                if let Some((_, mut queued_task)) = self.waiting_queue.remove(&dep_task_id) {
                    // Update task status
                    queued_task.task.status = TaskStatus::Failed;
                    queued_task.task.error = Some(format!("Dependency {} was cancelled", task_id));
                    
                    // Recursively cancel dependent tasks
                    let _ = self.cancel_task(&dep_task_id).await;
                }
            }
            
            // Remove from dependency graph
            self.dependency_graph.remove(task_id);
            
            return Ok(Some(queued_task.task));
        }
        
        // Check if task is in ready queue
        {
            let mut queue = self.ready_queue.lock().await;
            let mut new_queue = BinaryHeap::new();
            let mut found_task = None;
            
            // Rebuild queue without the cancelled task
            while let Some(Reverse(task)) = queue.pop() {
                if task.task.id == *task_id {
                    found_task = Some(task.task);
                } else {
                    new_queue.push(Reverse(task));
                }
            }
            
            *queue = new_queue;
            
            if let Some(task) = found_task {
                // Get dependent tasks
                let dependent_tasks = self.get_dependent_tasks(task_id);
                
                // Mark dependent tasks as failed
                for dep_task_id in dependent_tasks {
                    if let Some((_, mut queued_task)) = self.waiting_queue.remove(&dep_task_id) {
                        // Update task status
                        queued_task.task.status = TaskStatus::Failed;
                        queued_task.task.error = Some(format!("Dependency {} was cancelled", task_id));
                        
                        // Recursively cancel dependent tasks
                        let _ = self.cancel_task(&dep_task_id).await;
                    }
                }
                
                // Remove from dependency graph
                self.dependency_graph.remove(task_id);
                
                return Ok(Some(task));
            }
        }
        
        Ok(None)
    }
    
    /// Gets the size of the queue
    pub fn get_queue_size(&self) -> usize {
        let ready_size = match self.ready_queue.try_lock() {
            Ok(queue) => queue.len(),
            Err(_) => 0,
        };
        
        let waiting_size = self.waiting_queue.len();
        
        ready_size + waiting_size
    }
    
    /// Gets the number of ready tasks
    pub fn get_ready_queue_size(&self) -> usize {
        match self.ready_queue.try_lock() {
            Ok(queue) => queue.len(),
            Err(_) => 0,
        }
    }
    
    /// Gets the number of waiting tasks
    pub fn get_waiting_queue_size(&self) -> usize {
        self.waiting_queue.len()
    }
    
    /// Checks if a task is queued
    pub fn is_task_queued(&self, task_id: &TaskId) -> bool {
        // Check waiting queue
        if self.waiting_queue.contains_key(task_id) {
            return true;
        }
        
        // Check ready queue
        match self.ready_queue.try_lock() {
            Ok(queue) => {
                for Reverse(task) in queue.iter() {
                    if task.task.id == *task_id {
                        return true;
                    }
                }
                false
            },
            Err(_) => false,
        }
    }
    
    /// Checks if a dependency is satisfied
    fn is_dependency_satisfied(&self, task_id: &TaskId) -> bool {
        // In a real implementation, we would check if the task is completed
        // For now, we just assume it's not satisfied if it's in any queue
        !self.is_task_queued(task_id)
    }
    
    /// Gets tasks that depend on a given task
    fn get_dependent_tasks(&self, task_id: &TaskId) -> HashSet<TaskId> {
        match self.dependency_graph.get(task_id) {
            Some(deps) => deps.clone(),
            None => HashSet::new(),
        }
    }
    
    /// Gets all queued tasks
    pub async fn get_all_queued_tasks(&self) -> Vec<Task> {
        let mut tasks = Vec::new();
        
        // Get tasks from ready queue
        {
            let queue = self.ready_queue.lock().await;
            for Reverse(task) in queue.iter() {
                tasks.push(task.task.clone());
            }
        }
        
        // Get tasks from waiting queue
        for entry in self.waiting_queue.iter() {
            tasks.push(entry.task.clone());
        }
        
        tasks
    }
    
    /// Gets a queued task by ID
    pub async fn get_queued_task(&self, task_id: &TaskId) -> Option<Task> {
        // Check waiting queue
        if let Some(entry) = self.waiting_queue.get(task_id) {
            return Some(entry.task.clone());
        }
        
        // Check ready queue
        let queue = self.ready_queue.lock().await;
        for Reverse(task) in queue.iter() {
            if task.task.id == *task_id {
                return Some(task.task.clone());
            }
        }
        
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    // These tests would require mocking the dependencies
    // For now, we just have placeholder tests
    
    #[test]
    fn test_task_queue_manager() {
        // Placeholder test
        assert!(true);
    }
}

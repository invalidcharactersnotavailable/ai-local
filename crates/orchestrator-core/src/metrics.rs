//! Metrics collection and reporting for the orchestrator
//!
//! This module provides functionality for collecting and reporting metrics
//! about the AI Orchestrator's performance and resource usage.

use std::sync::Arc;
use tokio::sync::RwLock;
use anyhow::Result;
use tracing::{info, warn, error, debug, trace};
use std::collections::HashMap;
use std::time::{Duration, Instant};

use common::error::Error;
use common::models::{SystemResources, Task};
use common::types::PerformanceMetrics;
use performance_monitor::MetricsCollector;

/// Metrics manager for the orchestrator
pub struct MetricsManager {
    /// Performance metrics
    metrics: Arc<RwLock<PerformanceMetrics>>,
    
    /// Metrics collector
    metrics_collector: Arc<MetricsCollector>,
    
    /// Last update time
    last_update: Arc<RwLock<Instant>>,
    
    /// Update interval
    update_interval: Duration,
    
    /// Task execution times (task_id -> execution time)
    task_execution_times: Arc<RwLock<HashMap<String, Duration>>>,
}

impl MetricsManager {
    /// Creates a new metrics manager
    pub fn new(metrics_collector: Arc<MetricsCollector>) -> Result<Self> {
        Ok(Self {
            metrics: Arc::new(RwLock::new(PerformanceMetrics {
                cpu_usage_percent: 0.0,
                memory_usage_bytes: 0,
                gpu_usage_percent: None,
                gpu_memory_usage_bytes: None,
                disk_usage_bytes: 0,
                network_usage_bytes_per_second: 0,
                inference_latency_ms: 0.0,
                inference_throughput_rps: 0.0,
                queue_length: 0,
                active_tasks_count: 0,
            })),
            metrics_collector,
            last_update: Arc::new(RwLock::new(Instant::now())),
            update_interval: Duration::from_secs(1),
            task_execution_times: Arc::new(RwLock::new(HashMap::new())),
        })
    }
    
    /// Starts the metrics manager
    pub async fn start(&self) -> Result<()> {
        info!("Starting metrics manager");
        
        // Start metrics collection
        self.start_metrics_collection().await?;
        
        info!("Metrics manager started successfully");
        
        Ok(())
    }
    
    /// Stops the metrics manager
    pub async fn stop(&self) -> Result<()> {
        info!("Stopping metrics manager");
        
        // Stop metrics collection is handled by the metrics collector
        
        info!("Metrics manager stopped successfully");
        
        Ok(())
    }
    
    /// Starts metrics collection
    async fn start_metrics_collection(&self) -> Result<()> {
        let metrics = self.metrics.clone();
        let metrics_collector = self.metrics_collector.clone();
        let last_update = self.last_update.clone();
        let update_interval = self.update_interval;
        
        tokio::spawn(async move {
            loop {
                // Check if it's time to update metrics
                let now = Instant::now();
                let should_update = {
                    let last = last_update.read().await;
                    now.duration_since(*last) >= update_interval
                };
                
                if should_update {
                    // Update last update time
                    {
                        let mut last = last_update.write().await;
                        *last = now;
                    }
                    
                    // Collect metrics
                    if let Ok(collected_metrics) = metrics_collector.collect().await {
                        // Update metrics
                        let mut metrics_guard = metrics.write().await;
                        *metrics_guard = collected_metrics;
                    }
                }
                
                // Sleep for a short time
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
        });
        
        Ok(())
    }
    
    /// Gets the current performance metrics
    pub async fn get_metrics(&self) -> Result<PerformanceMetrics> {
        let metrics = self.metrics.read().await;
        Ok(metrics.clone())
    }
    
    /// Records the start of a task execution
    pub async fn record_task_start(&self, task_id: &str) -> Result<()> {
        let mut task_times = self.task_execution_times.write().await;
        task_times.insert(task_id.to_string(), Duration::from_secs(0));
        
        // Update active tasks count
        {
            let mut metrics = self.metrics.write().await;
            metrics.active_tasks_count = task_times.len();
        }
        
        Ok(())
    }
    
    /// Records the completion of a task execution
    pub async fn record_task_completion(&self, task_id: &str, duration: Duration) -> Result<()> {
        let mut task_times = self.task_execution_times.write().await;
        task_times.insert(task_id.to_string(), duration);
        
        // Calculate average latency
        let total_tasks = task_times.len();
        let total_duration: Duration = task_times.values().sum();
        let avg_duration = if total_tasks > 0 {
            total_duration.as_secs_f64() / total_tasks as f64
        } else {
            0.0
        };
        
        // Update metrics
        {
            let mut metrics = self.metrics.write().await;
            metrics.inference_latency_ms = avg_duration * 1000.0;
            metrics.active_tasks_count = total_tasks;
            
            // Calculate throughput (tasks per second)
            if avg_duration > 0.0 {
                metrics.inference_throughput_rps = 1.0 / avg_duration;
            } else {
                metrics.inference_throughput_rps = 0.0;
            }
        }
        
        Ok(())
    }
    
    /// Records a task failure
    pub async fn record_task_failure(&self, task_id: &str) -> Result<()> {
        let mut task_times = self.task_execution_times.write().await;
        task_times.remove(task_id);
        
        // Update active tasks count
        {
            let mut metrics = self.metrics.write().await;
            metrics.active_tasks_count = task_times.len();
        }
        
        Ok(())
    }
    
    /// Updates the queue length
    pub async fn update_queue_length(&self, length: usize) -> Result<()> {
        let mut metrics = self.metrics.write().await;
        metrics.queue_length = length;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_metrics_recording() {
        // This is a placeholder for a real test
        // In a real implementation, we would use mocks for the dependencies
        assert!(true);
    }
}

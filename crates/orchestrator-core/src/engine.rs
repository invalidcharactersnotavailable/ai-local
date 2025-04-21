//! Core orchestration engine implementation
//!
//! This module provides the central orchestration engine for the AI Orchestrator,
//! coordinating all components and managing the system's lifecycle.

use std::sync::Arc;
use tokio::sync::{RwLock, Mutex, mpsc};
use parking_lot::RwLock as ParkingLotRwLock;
use dashmap::DashMap;
use anyhow::Result;
use tracing::{info, warn, error, debug, trace};
use async_trait::async_trait;

use common::error::Error;
use common::models::{Model, Task, SystemResources};
use common::types::OperationalMode;
use model_manager::ModelRepository;
use resource_manager::ResourceMonitor;
use task_scheduler::TaskQueue;
use hardware_profiler::SystemProfiler;
use performance_monitor::MetricsCollector;
use scaling_adapter::DynamicScaler;
use config::ConfigManager;
use logging::Logger;

use crate::state::OrchestratorState;
use crate::coordinator::Coordinator;
use crate::lifecycle::LifecycleManager;

/// The main orchestration engine
pub struct OrchestratorEngine {
    /// Current state of the orchestrator
    state: Arc<RwLock<OrchestratorState>>,
    
    /// Model repository
    model_repository: Arc<ModelRepository>,
    
    /// Resource monitor
    resource_monitor: Arc<ResourceMonitor>,
    
    /// Task queue
    task_queue: Arc<TaskQueue>,
    
    /// System profiler
    system_profiler: Arc<SystemProfiler>,
    
    /// Metrics collector
    metrics_collector: Arc<MetricsCollector>,
    
    /// Dynamic scaler
    dynamic_scaler: Arc<DynamicScaler>,
    
    /// Configuration manager
    config_manager: Arc<ConfigManager>,
    
    /// Logger
    logger: Arc<Logger>,
    
    /// Lifecycle manager
    lifecycle_manager: Arc<LifecycleManager>,
    
    /// Coordinator
    coordinator: Arc<Coordinator>,
    
    /// Active models
    active_models: Arc<DashMap<String, Arc<Model>>>,
    
    /// Active tasks
    active_tasks: Arc<DashMap<String, Arc<Task>>>,
    
    /// Shutdown signal sender
    shutdown_tx: mpsc::Sender<()>,
    
    /// Shutdown signal receiver
    shutdown_rx: Mutex<mpsc::Receiver<()>>,
    
    /// Operational mode
    operational_mode: Arc<ParkingLotRwLock<OperationalMode>>,
}

impl OrchestratorEngine {
    /// Creates a new orchestration engine
    pub async fn new(config_path: Option<String>) -> Result<Self> {
        // Initialize configuration
        let config_manager = Arc::new(ConfigManager::new(config_path)?);
        
        // Initialize logger
        let logger = Arc::new(Logger::new(config_manager.clone())?);
        
        // Initialize system profiler
        let system_profiler = Arc::new(SystemProfiler::new()?);
        
        // Initialize resource monitor
        let resource_monitor = Arc::new(ResourceMonitor::new(
            system_profiler.clone(),
            config_manager.clone(),
        )?);
        
        // Initialize model repository
        let model_repository = Arc::new(ModelRepository::new(
            config_manager.clone(),
            resource_monitor.clone(),
        )?);
        
        // Initialize task queue
        let task_queue = Arc::new(TaskQueue::new(
            config_manager.clone(),
        )?);
        
        // Initialize metrics collector
        let metrics_collector = Arc::new(MetricsCollector::new(
            config_manager.clone(),
            resource_monitor.clone(),
        )?);
        
        // Initialize dynamic scaler
        let dynamic_scaler = Arc::new(DynamicScaler::new(
            resource_monitor.clone(),
            metrics_collector.clone(),
            config_manager.clone(),
        )?);
        
        // Initialize state
        let state = Arc::new(RwLock::new(OrchestratorState::new()));
        
        // Initialize lifecycle manager
        let lifecycle_manager = Arc::new(LifecycleManager::new(
            state.clone(),
            config_manager.clone(),
        )?);
        
        // Initialize coordinator
        let coordinator = Arc::new(Coordinator::new(
            state.clone(),
            model_repository.clone(),
            resource_monitor.clone(),
            task_queue.clone(),
            dynamic_scaler.clone(),
            config_manager.clone(),
        )?);
        
        // Initialize active models and tasks
        let active_models = Arc::new(DashMap::new());
        let active_tasks = Arc::new(DashMap::new());
        
        // Initialize shutdown channel
        let (shutdown_tx, shutdown_rx) = mpsc::channel(1);
        
        // Initialize operational mode
        let operational_mode = Arc::new(ParkingLotRwLock::new(OperationalMode::Standard));
        
        Ok(Self {
            state,
            model_repository,
            resource_monitor,
            task_queue,
            system_profiler,
            metrics_collector,
            dynamic_scaler,
            config_manager,
            logger,
            lifecycle_manager,
            coordinator,
            active_models,
            active_tasks,
            shutdown_tx,
            shutdown_rx: Mutex::new(shutdown_rx),
            operational_mode,
        })
    }
    
    /// Starts the orchestration engine
    pub async fn start(&self) -> Result<()> {
        info!("Starting AI Orchestrator engine");
        
        // Start the lifecycle manager
        self.lifecycle_manager.start().await?;
        
        // Start the resource monitor
        self.resource_monitor.start().await?;
        
        // Start the metrics collector
        self.metrics_collector.start().await?;
        
        // Start the coordinator
        self.coordinator.start().await?;
        
        // Update state to running
        let mut state = self.state.write().await;
        *state = OrchestratorState::Running;
        drop(state);
        
        info!("AI Orchestrator engine started successfully");
        
        Ok(())
    }
    
    /// Stops the orchestration engine
    pub async fn stop(&self) -> Result<()> {
        info!("Stopping AI Orchestrator engine");
        
        // Update state to stopping
        let mut state = self.state.write().await;
        *state = OrchestratorState::Stopping;
        drop(state);
        
        // Stop the coordinator
        self.coordinator.stop().await?;
        
        // Stop the metrics collector
        self.metrics_collector.stop().await?;
        
        // Stop the resource monitor
        self.resource_monitor.stop().await?;
        
        // Stop the lifecycle manager
        self.lifecycle_manager.stop().await?;
        
        // Update state to stopped
        let mut state = self.state.write().await;
        *state = OrchestratorState::Stopped;
        drop(state);
        
        info!("AI Orchestrator engine stopped successfully");
        
        Ok(())
    }
    
    /// Shuts down the orchestration engine
    pub async fn shutdown(&self) -> Result<()> {
        info!("Shutting down AI Orchestrator engine");
        
        // Stop all components
        self.stop().await?;
        
        // Send shutdown signal
        let _ = self.shutdown_tx.send(()).await;
        
        info!("AI Orchestrator engine shutdown complete");
        
        Ok(())
    }
    
    /// Waits for shutdown signal
    pub async fn wait_for_shutdown(&self) -> Result<()> {
        let mut rx = self.shutdown_rx.lock().await;
        let _ = rx.recv().await;
        Ok(())
    }
    
    /// Gets the current state of the orchestrator
    pub async fn get_state(&self) -> OrchestratorState {
        let state = self.state.read().await;
        state.clone()
    }
    
    /// Gets the current system resources
    pub async fn get_system_resources(&self) -> Result<SystemResources> {
        self.resource_monitor.get_system_resources().await
    }
    
    /// Gets the current operational mode
    pub fn get_operational_mode(&self) -> OperationalMode {
        *self.operational_mode.read()
    }
    
    /// Sets the operational mode
    pub async fn set_operational_mode(&self, mode: OperationalMode) -> Result<()> {
        info!("Changing operational mode to: {:?}", mode);
        
        // Update the operational mode
        {
            let mut current_mode = self.operational_mode.write();
            *current_mode = mode;
        }
        
        // Notify the coordinator of the mode change
        self.coordinator.on_mode_change(mode).await?;
        
        // Adjust resource allocation based on new mode
        self.dynamic_scaler.adjust_for_mode(mode).await?;
        
        info!("Operational mode changed to: {:?}", mode);
        
        Ok(())
    }
    
    /// Loads a model
    pub async fn load_model(&self, model_id: &str) -> Result<Arc<Model>> {
        // Check if model is already loaded
        if let Some(model) = self.active_models.get(model_id) {
            return Ok(model.clone());
        }
        
        // Load the model
        let model = self.model_repository.load_model(model_id).await?;
        
        // Add to active models
        let model_arc = Arc::new(model);
        self.active_models.insert(model_id.to_string(), model_arc.clone());
        
        Ok(model_arc)
    }
    
    /// Unloads a model
    pub async fn unload_model(&self, model_id: &str) -> Result<()> {
        // Check if model is loaded
        if !self.active_models.contains_key(model_id) {
            return Err(Error::NotFound(format!("Model not loaded: {}", model_id)).into());
        }
        
        // Unload the model
        self.model_repository.unload_model(model_id).await?;
        
        // Remove from active models
        self.active_models.remove(model_id);
        
        Ok(())
    }
    
    /// Submits a task
    pub async fn submit_task(&self, task: Task) -> Result<String> {
        // Submit the task to the queue
        let task_id = self.task_queue.submit(task).await?;
        
        // Notify the coordinator of the new task
        self.coordinator.on_task_submitted(&task_id).await?;
        
        Ok(task_id)
    }
    
    /// Cancels a task
    pub async fn cancel_task(&self, task_id: &str) -> Result<()> {
        // Cancel the task
        self.task_queue.cancel(task_id).await?;
        
        // Notify the coordinator of the cancelled task
        self.coordinator.on_task_cancelled(task_id).await?;
        
        Ok(())
    }
    
    /// Gets a task by ID
    pub async fn get_task(&self, task_id: &str) -> Result<Arc<Task>> {
        // Check active tasks first
        if let Some(task) = self.active_tasks.get(task_id) {
            return Ok(task.clone());
        }
        
        // Get from task queue
        let task = self.task_queue.get(task_id).await?;
        Ok(Arc::new(task))
    }
    
    /// Lists all tasks
    pub async fn list_tasks(&self) -> Result<Vec<Arc<Task>>> {
        self.task_queue.list().await
    }
    
    /// Lists all models
    pub async fn list_models(&self) -> Result<Vec<Arc<Model>>> {
        self.model_repository.list_models().await
    }
    
    /// Gets a model by ID
    pub async fn get_model(&self, model_id: &str) -> Result<Arc<Model>> {
        // Check active models first
        if let Some(model) = self.active_models.get(model_id) {
            return Ok(model.clone());
        }
        
        // Get from model repository
        let model = self.model_repository.get_model(model_id).await?;
        Ok(Arc::new(model))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_engine_lifecycle() {
        // Create engine
        let engine = OrchestratorEngine::new(None).await.unwrap();
        
        // Start engine
        engine.start().await.unwrap();
        
        // Check state
        assert_eq!(engine.get_state().await, OrchestratorState::Running);
        
        // Stop engine
        engine.stop().await.unwrap();
        
        // Check state
        assert_eq!(engine.get_state().await, OrchestratorState::Stopped);
    }
    
    #[tokio::test]
    async fn test_operational_mode() {
        // Create engine
        let engine = OrchestratorEngine::new(None).await.unwrap();
        
        // Check default mode
        assert_eq!(engine.get_operational_mode(), OperationalMode::Standard);
        
        // Change mode
        engine.set_operational_mode(OperationalMode::HighEffort).await.unwrap();
        
        // Check new mode
        assert_eq!(engine.get_operational_mode(), OperationalMode::HighEffort);
    }
}

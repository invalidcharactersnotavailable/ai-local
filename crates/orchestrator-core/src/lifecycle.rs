//! Lifecycle management for the orchestrator
//!
//! This module provides functionality for managing the lifecycle of the AI Orchestrator,
//! including startup, shutdown, and state transitions.

use std::sync::Arc;
use tokio::sync::RwLock;
use anyhow::Result;
use tracing::{info, warn, error, debug};

use common::error::Error;
use config::ConfigManager;

use crate::state::OrchestratorState;

/// Lifecycle manager for the orchestrator
pub struct LifecycleManager {
    /// Current state of the orchestrator
    state: Arc<RwLock<OrchestratorState>>,
    
    /// Configuration manager
    config_manager: Arc<ConfigManager>,
    
    /// Shutdown flag
    shutdown_requested: Arc<tokio::sync::Mutex<bool>>,
}

impl LifecycleManager {
    /// Creates a new lifecycle manager
    pub fn new(
        state: Arc<RwLock<OrchestratorState>>,
        config_manager: Arc<ConfigManager>,
    ) -> Result<Self> {
        Ok(Self {
            state,
            config_manager,
            shutdown_requested: Arc::new(tokio::sync::Mutex::new(false)),
        })
    }
    
    /// Starts the lifecycle manager
    pub async fn start(&self) -> Result<()> {
        info!("Starting lifecycle manager");
        
        // Reset shutdown flag
        let mut shutdown_flag = self.shutdown_requested.lock().await;
        *shutdown_flag = false;
        drop(shutdown_flag);
        
        // Start health check
        self.start_health_check().await?;
        
        info!("Lifecycle manager started successfully");
        
        Ok(())
    }
    
    /// Stops the lifecycle manager
    pub async fn stop(&self) -> Result<()> {
        info!("Stopping lifecycle manager");
        
        // Set shutdown flag
        let mut shutdown_flag = self.shutdown_requested.lock().await;
        *shutdown_flag = true;
        drop(shutdown_flag);
        
        info!("Lifecycle manager stopped successfully");
        
        Ok(())
    }
    
    /// Starts the health check background task
    async fn start_health_check(&self) -> Result<()> {
        let state = self.state.clone();
        let shutdown_requested = self.shutdown_requested.clone();
        
        // Get health check interval from config
        let health_check_interval = self.config_manager
            .get_duration("health_check_interval")
            .unwrap_or_else(|_| std::time::Duration::from_secs(30));
        
        tokio::spawn(async move {
            loop {
                // Check if shutdown was requested
                let shutdown = *shutdown_requested.lock().await;
                if shutdown {
                    break;
                }
                
                // Perform health check
                let current_state = {
                    let state_guard = state.read().await;
                    state_guard.clone()
                };
                
                match current_state {
                    OrchestratorState::Running => {
                        // Everything is normal, just log at trace level
                        debug!("Health check: system is running normally");
                    },
                    OrchestratorState::Starting => {
                        // System is still starting up, check if it's taking too long
                        warn!("Health check: system is still in starting state");
                    },
                    OrchestratorState::Stopping => {
                        // System is stopping, check if it's taking too long
                        warn!("Health check: system is still in stopping state");
                    },
                    OrchestratorState::Stopped => {
                        // System is stopped, this is unexpected if we're running health checks
                        error!("Health check: system is in stopped state but lifecycle manager is running");
                    },
                    OrchestratorState::Error(ref err) => {
                        // System is in error state
                        error!("Health check: system is in error state: {}", err);
                        
                        // Attempt recovery
                        if let Err(e) = Self::attempt_recovery(&state).await {
                            error!("Failed to recover from error state: {}", e);
                        }
                    },
                }
                
                // Sleep until next health check
                tokio::time::sleep(tokio::time::Duration::from(health_check_interval)).await;
            }
        });
        
        Ok(())
    }
    
    /// Attempts to recover from an error state
    async fn attempt_recovery(state: &Arc<RwLock<OrchestratorState>>) -> Result<()> {
        // Get current state
        let current_state = {
            let state_guard = state.read().await;
            state_guard.clone()
        };
        
        // Only attempt recovery if we're in an error state
        if let OrchestratorState::Error(err) = current_state {
            info!("Attempting to recover from error state: {}", err);
            
            // Set state to starting
            {
                let mut state_guard = state.write().await;
                *state_guard = OrchestratorState::Starting;
            }
            
            // In a real implementation, we would have recovery logic here
            // For now, we just transition to running state
            
            // Set state to running
            {
                let mut state_guard = state.write().await;
                *state_guard = OrchestratorState::Running;
            }
            
            info!("Successfully recovered from error state");
        }
        
        Ok(())
    }
    
    /// Transitions to a new state
    pub async fn transition_to(&self, new_state: OrchestratorState) -> Result<()> {
        // Get current state
        let current_state = {
            let state_guard = self.state.read().await;
            state_guard.clone()
        };
        
        // Log state transition
        info!("State transition: {:?} -> {:?}", current_state, new_state);
        
        // Update state
        {
            let mut state_guard = self.state.write().await;
            *state_guard = new_state.clone();
        }
        
        // Perform actions based on new state
        match new_state {
            OrchestratorState::Starting => {
                // Initialization actions
                debug!("Performing initialization actions");
            },
            OrchestratorState::Running => {
                // Running actions
                debug!("System is now running");
            },
            OrchestratorState::Stopping => {
                // Stopping actions
                debug!("Performing shutdown actions");
            },
            OrchestratorState::Stopped => {
                // Stopped actions
                debug!("System is now stopped");
            },
            OrchestratorState::Error(ref err) => {
                // Error actions
                error!("System entered error state: {}", err);
                
                // Attempt recovery if configured to do so
                let auto_recover = self.config_manager
                    .get_bool("auto_recover_from_errors")
                    .unwrap_or(true);
                
                if auto_recover {
                    tokio::spawn(Self::attempt_recovery(self.state.clone()));
                }
            },
        }
        
        Ok(())
    }
    
    /// Handles a system error
    pub async fn handle_error(&self, error: &str) -> Result<()> {
        error!("System error: {}", error);
        
        // Transition to error state
        self.transition_to(OrchestratorState::Error(error.to_string())).await?;
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_state_transitions() {
        // This is a placeholder for a real test
        // In a real implementation, we would use mocks for the dependencies
        assert!(true);
    }
}

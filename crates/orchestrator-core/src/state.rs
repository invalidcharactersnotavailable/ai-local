//! State management for the orchestrator
//!
//! This module provides the state representation for the AI Orchestrator,
//! tracking the current operational state of the system.

use std::fmt;
use serde::{Serialize, Deserialize};

/// Represents the current state of the orchestrator
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrchestratorState {
    /// System is starting up
    Starting,
    
    /// System is running normally
    Running,
    
    /// System is in the process of stopping
    Stopping,
    
    /// System is stopped
    Stopped,
    
    /// System is in an error state
    Error(String),
}

impl OrchestratorState {
    /// Creates a new orchestrator state
    pub fn new() -> Self {
        OrchestratorState::Starting
    }
    
    /// Returns true if the system is running
    pub fn is_running(&self) -> bool {
        matches!(self, OrchestratorState::Running)
    }
    
    /// Returns true if the system is starting
    pub fn is_starting(&self) -> bool {
        matches!(self, OrchestratorState::Starting)
    }
    
    /// Returns true if the system is stopping
    pub fn is_stopping(&self) -> bool {
        matches!(self, OrchestratorState::Stopping)
    }
    
    /// Returns true if the system is stopped
    pub fn is_stopped(&self) -> bool {
        matches!(self, OrchestratorState::Stopped)
    }
    
    /// Returns true if the system is in an error state
    pub fn is_error(&self) -> bool {
        matches!(self, OrchestratorState::Error(_))
    }
    
    /// Gets the error message if in error state
    pub fn error_message(&self) -> Option<&str> {
        match self {
            OrchestratorState::Error(msg) => Some(msg),
            _ => None,
        }
    }
}

impl fmt::Display for OrchestratorState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OrchestratorState::Starting => write!(f, "Starting"),
            OrchestratorState::Running => write!(f, "Running"),
            OrchestratorState::Stopping => write!(f, "Stopping"),
            OrchestratorState::Stopped => write!(f, "Stopped"),
            OrchestratorState::Error(msg) => write!(f, "Error: {}", msg),
        }
    }
}

impl Default for OrchestratorState {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_state_methods() {
        let starting = OrchestratorState::Starting;
        assert!(starting.is_starting());
        assert!(!starting.is_running());
        assert!(!starting.is_stopping());
        assert!(!starting.is_stopped());
        assert!(!starting.is_error());
        assert_eq!(starting.error_message(), None);
        
        let running = OrchestratorState::Running;
        assert!(!running.is_starting());
        assert!(running.is_running());
        assert!(!running.is_stopping());
        assert!(!running.is_stopped());
        assert!(!running.is_error());
        assert_eq!(running.error_message(), None);
        
        let stopping = OrchestratorState::Stopping;
        assert!(!stopping.is_starting());
        assert!(!stopping.is_running());
        assert!(stopping.is_stopping());
        assert!(!stopping.is_stopped());
        assert!(!stopping.is_error());
        assert_eq!(stopping.error_message(), None);
        
        let stopped = OrchestratorState::Stopped;
        assert!(!stopped.is_starting());
        assert!(!stopped.is_running());
        assert!(!stopped.is_stopping());
        assert!(stopped.is_stopped());
        assert!(!stopped.is_error());
        assert_eq!(stopped.error_message(), None);
        
        let error = OrchestratorState::Error("Test error".to_string());
        assert!(!error.is_starting());
        assert!(!error.is_running());
        assert!(!error.is_stopping());
        assert!(!error.is_stopped());
        assert!(error.is_error());
        assert_eq!(error.error_message(), Some("Test error"));
    }
    
    #[test]
    fn test_display() {
        assert_eq!(OrchestratorState::Starting.to_string(), "Starting");
        assert_eq!(OrchestratorState::Running.to_string(), "Running");
        assert_eq!(OrchestratorState::Stopping.to_string(), "Stopping");
        assert_eq!(OrchestratorState::Stopped.to_string(), "Stopped");
        assert_eq!(OrchestratorState::Error("Test error".to_string()).to_string(), "Error: Test error");
    }
    
    #[test]
    fn test_default() {
        assert_eq!(OrchestratorState::default(), OrchestratorState::Starting);
    }
}

//! Error types for the common crate
//!
//! This module defines the common error types used throughout the AI Orchestrator system.

use std::fmt;
use thiserror::Error;

/// Result type for AI Orchestrator operations
pub type Result<T> = std::result::Result<T, Error>;

/// Common error type for AI Orchestrator operations
#[derive(Error, Debug)]
pub enum Error {
    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Serialization error
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// Configuration error
    #[error("Configuration error: {0}")]
    Config(String),

    /// Resource error
    #[error("Resource error: {0}")]
    Resource(String),

    /// Model error
    #[error("Model error: {0}")]
    Model(String),

    /// Task error
    #[error("Task error: {0}")]
    Task(String),

    /// Inference error
    #[error("Inference error: {0}")]
    Inference(String),

    /// Hardware error
    #[error("Hardware error: {0}")]
    Hardware(String),

    /// Authentication error
    #[error("Authentication error: {0}")]
    Authentication(String),

    /// Authorization error
    #[error("Authorization error: {0}")]
    Authorization(String),

    /// Not found error
    #[error("Not found: {0}")]
    NotFound(String),

    /// Already exists error
    #[error("Already exists: {0}")]
    AlreadyExists(String),

    /// Invalid argument error
    #[error("Invalid argument: {0}")]
    InvalidArgument(String),

    /// Timeout error
    #[error("Timeout: {0}")]
    Timeout(String),

    /// Internal error
    #[error("Internal error: {0}")]
    Internal(String),

    /// External service error
    #[error("External service error: {0}")]
    ExternalService(String),

    /// Unsupported operation error
    #[error("Unsupported operation: {0}")]
    UnsupportedOperation(String),
}

impl Error {
    /// Returns true if the error is a not found error
    pub fn is_not_found(&self) -> bool {
        matches!(self, Error::NotFound(_))
    }

    /// Returns true if the error is an already exists error
    pub fn is_already_exists(&self) -> bool {
        matches!(self, Error::AlreadyExists(_))
    }

    /// Returns true if the error is an authentication error
    pub fn is_authentication(&self) -> bool {
        matches!(self, Error::Authentication(_))
    }

    /// Returns true if the error is an authorization error
    pub fn is_authorization(&self) -> bool {
        matches!(self, Error::Authorization(_))
    }

    /// Returns true if the error is a timeout error
    pub fn is_timeout(&self) -> bool {
        matches!(self, Error::Timeout(_))
    }
}

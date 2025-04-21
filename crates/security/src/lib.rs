//! Authentication and authorization for AI Orchestrator
//!
//! This crate provides functionality for user authentication and authorization,
//! ensuring secure access to the AI Orchestrator.

pub mod auth;
pub mod users;
pub mod roles;
pub mod permissions;
pub mod tokens;
pub mod encryption;
pub mod audit;

// Re-export commonly used types
pub use auth::{Authenticator, Authorizer};
pub use users::UserManager;
pub use roles::RoleManager;
pub use permissions::PermissionManager;
pub use tokens::TokenManager;

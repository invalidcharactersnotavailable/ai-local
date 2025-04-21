//! External API implementation for AI Orchestrator
//!
//! This crate provides RESTful and WebSocket APIs for external applications
//! to interact with the AI Orchestrator.

pub mod rest;
pub mod websocket;
pub mod routes;
pub mod handlers;
pub mod middleware;
pub mod validation;
pub mod documentation;

// Re-export commonly used types
pub use rest::RestAPI;
pub use websocket::WebSocketAPI;
pub use routes::APIRoutes;
pub use middleware::APIMiddleware;

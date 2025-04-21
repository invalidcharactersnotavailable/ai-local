//! Command-line interface for AI Orchestrator
//!
//! This crate provides a user-friendly command-line interface for managing
//! the AI Orchestrator, including an interactive menu system.

pub mod cli;
pub mod menu;
pub mod commands;
pub mod ui;
pub mod system_detection;
pub mod interactive;
pub mod formatters;

// Re-export commonly used types
pub use cli::CLI;
pub use menu::MenuSystem;
pub use commands::CommandManager;
pub use ui::UIComponents;

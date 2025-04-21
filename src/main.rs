use std::sync::Arc;
use anyhow::Result;
use tokio;

use ai_orchestrator::AIOrchestrator;
use common::models::{TaskType, TaskPriority};
use task_scheduler::TaskDefinition;
use search_engine::{SearchEngine, SearchQuery, SortDirection, InMemorySearchDataSource};
use advanced_reasoning::{AdvancedReasoningEngine, ReasoningTask, ReasoningComplexity};
use web_interface::WebInterface;
use hardware_profiler::HardwareCapabilities;

#[tokio::main]
async fn main() -> Result<()> {
    println!("Starting AI Orchestrator with Environment Check...");
    
    // First, perform environment check
    println!("Performing environment check...");
    let hardware_profiler = Arc::new(HardwareCapabilities::new()?);
    
    // Display hardware information
    let cpu_info = hardware_profiler.get_cpu_info()?;
    println!("CPU: {} with {} physical cores, {} logical cores", 
             cpu_info.model, cpu_info.physical_cores, cpu_info.logical_cores);
    
    let memory_info = hardware_profiler.get_memory_info()?;
    println!("Memory: {:.2} GB total, {:.2} GB available", 
             memory_info.total_bytes as f64 / 1_073_741_824.0,
             memory_info.available_bytes as f64 / 1_073_741_824.0);
    
    let gpu_info = hardware_profiler.get_gpu_info()?;
    if !gpu_info.is_empty() {
        println!("GPUs detected:");
        for (i, gpu) in gpu_info.iter().enumerate() {
            println!("  GPU {}: {}, {:.2} GB memory", 
                     i, gpu.name, 
                     gpu.memory_bytes as f64 / 1_073_741_824.0);
        }
    } else {
        println!("No GPUs detected");
    }
    
    // Create and start the orchestrator
    println!("\nInitializing AI Orchestrator...");
    let orchestrator = AIOrchestrator::new().await?;
    orchestrator.start().await?;
    
    println!("AI Orchestrator started successfully!");
    
    // Initialize search engine
    println!("Initializing Search Engine...");
    let search_engine = Arc::new(SearchEngine::new(
        orchestrator.get_config_manager(),
        orchestrator.get_performance_optimizer(),
    ));
    
    // Create and register an in-memory data source
    let data_source = Arc::new(InMemorySearchDataSource::new("documents"));
    search_engine.register_data_source(data_source.clone()).await?;
    
    // Add some sample documents
    data_source.add_document(
        "doc1",
        "Introduction to AI",
        "Artificial Intelligence (AI) is the simulation of human intelligence processes by machines.",
        [("category".to_string(), "ai".to_string())].into_iter().collect(),
    ).await?;
    
    data_source.add_document(
        "doc2",
        "Machine Learning Basics",
        "Machine Learning is a subset of AI that enables systems to learn and improve from experience.",
        [("category".to_string(), "ml".to_string())].into_iter().collect(),
    ).await?;
    
    data_source.add_document(
        "doc3",
        "Neural Networks",
        "Neural networks are computing systems inspired by the biological neural networks in animal brains.",
        [("category".to_string(), "ml".to_string())].into_iter().collect(),
    ).await?;
    
    println!("Search Engine initialized with {} documents", 
             data_source.document_count().await?);
    
    // Initialize advanced reasoning engine
    println!("Initializing Advanced Reasoning Engine...");
    let reasoning_engine = Arc::new(AdvancedReasoningEngine::new(
        orchestrator.get_config_manager(),
        orchestrator.get_model_manager(),
        orchestrator.get_resource_manager(),
        orchestrator.get_performance_optimizer(),
    )?);
    
    println!("Advanced Reasoning Engine initialized");
    println!("Default complexity: {:?}", reasoning_engine.get_default_complexity());
    println!("Auto-scaling: {}", if reasoning_engine.is_auto_scaling_enabled() { "Enabled" } else { "Disabled" });
    
    // Initialize and start web interface
    println!("\nStarting Web Interface...");
    let web_interface = WebInterface::new(
        orchestrator.get_config_manager(),
        orchestrator.get_model_manager(),
        orchestrator.get_resource_manager(),
        orchestrator.get_task_scheduler(),
        orchestrator.get_performance_optimizer(),
        hardware_profiler,
        search_engine,
        reasoning_engine,
    )?;
    
    web_interface.start().await?;
    println!("Web Interface started successfully!");
    println!("A browser window should open automatically. If not, please navigate to http://localhost:80");
    
    // Keep the application running
    println!("\nAI Orchestrator is now running. Press Ctrl+C to stop.");
    
    // Wait for Ctrl+C
    tokio::signal::ctrl_c().await?;
    
    // Stop the orchestrator
    println!("\nStopping AI Orchestrator...");
    orchestrator.stop().await?;
    println!("AI Orchestrator stopped successfully!");
    
    Ok(())
}

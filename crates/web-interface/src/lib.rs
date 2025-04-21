//! Web interface for the AI orchestrator
//!
//! This module provides a web interface for interacting with the AI orchestrator,
//! including model recommendations and task management.

use std::sync::Arc;
use anyhow::Result;
use tracing::{info, warn, error, debug, trace};
use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use tokio::sync::RwLock;
use std::process::Command;
use std::time::Duration;

use actix_web::{web, App, HttpServer, HttpResponse, Responder};
use actix_web::middleware::Logger;
use actix_files::Files;
use actix_cors::Cors;

use common::error::Error;
use common::models::{Task, TaskId, TaskType, TaskStatus, TaskPriority};
use config::ConfigManager;
use model_manager::ModelManager;
use resource_manager::ResourceManager;
use task_scheduler::TaskScheduler;
use performance_optimizations::PerformanceOptimizer;
use hardware_profiler::HardwareCapabilities;
use search_engine::SearchEngine;
use advanced_reasoning::AdvancedReasoningEngine;

/// Model recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRecommendation {
    /// Model ID
    pub id: String,
    
    /// Model name
    pub name: String,
    
    /// Model description
    pub description: String,
    
    /// Model size in parameters
    pub parameters: u64,
    
    /// Model size in bytes
    pub size_bytes: u64,
    
    /// Minimum required memory in bytes
    pub min_memory_bytes: u64,
    
    /// Recommended memory in bytes
    pub recommended_memory_bytes: u64,
    
    /// Whether GPU is required
    pub requires_gpu: bool,
    
    /// Minimum required GPU memory in bytes
    pub min_gpu_memory_bytes: Option<u64>,
    
    /// Recommended GPU memory in bytes
    pub recommended_gpu_memory_bytes: Option<u64>,
    
    /// Compatibility score (0-100)
    pub compatibility_score: u8,
    
    /// Whether this model is recommended for the current system
    pub is_recommended: bool,
    
    /// Model tags
    pub tags: Vec<String>,
    
    /// Model capabilities
    pub capabilities: Vec<String>,
    
    /// Download URL
    pub download_url: String,
}

/// System status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemStatus {
    /// CPU information
    pub cpu: CpuInfo,
    
    /// Memory information
    pub memory: MemoryInfo,
    
    /// GPU information
    pub gpus: Vec<GpuInfo>,
    
    /// Task information
    pub tasks: TaskInfo,
    
    /// Model information
    pub models: ModelInfo,
}

/// CPU information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuInfo {
    /// CPU model
    pub model: String,
    
    /// Number of physical cores
    pub physical_cores: usize,
    
    /// Number of logical cores
    pub logical_cores: usize,
    
    /// CPU usage percentage
    pub usage_percent: f32,
}

/// Memory information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryInfo {
    /// Total memory in bytes
    pub total_bytes: u64,
    
    /// Available memory in bytes
    pub available_bytes: u64,
    
    /// Used memory in bytes
    pub used_bytes: u64,
    
    /// Memory usage percentage
    pub usage_percent: f32,
}

/// GPU information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuInfo {
    /// GPU name
    pub name: String,
    
    /// Total memory in bytes
    pub total_memory_bytes: u64,
    
    /// Available memory in bytes
    pub available_memory_bytes: u64,
    
    /// GPU usage percentage
    pub usage_percent: f32,
}

/// Task information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskInfo {
    /// Number of running tasks
    pub running: usize,
    
    /// Number of queued tasks
    pub queued: usize,
    
    /// Number of completed tasks
    pub completed: usize,
}

/// Model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Number of available models
    pub available: usize,
    
    /// Number of loaded models
    pub loaded: usize,
    
    /// Total size of all models in bytes
    pub total_size_bytes: u64,
}

/// Web interface for the AI orchestrator
pub struct WebInterface {
    /// Configuration manager
    config_manager: Arc<ConfigManager>,
    
    /// Model manager
    model_manager: Arc<ModelManager>,
    
    /// Resource manager
    resource_manager: Arc<ResourceManager>,
    
    /// Task scheduler
    task_scheduler: Arc<TaskScheduler>,
    
    /// Performance optimizer
    performance_optimizer: Arc<PerformanceOptimizer>,
    
    /// Hardware profiler
    hardware_profiler: Arc<HardwareCapabilities>,
    
    /// Search engine
    search_engine: Arc<SearchEngine>,
    
    /// Advanced reasoning engine
    reasoning_engine: Arc<AdvancedReasoningEngine>,
    
    /// Server address
    address: String,
    
    /// Server port
    port: u16,
    
    /// Model recommendations
    model_recommendations: RwLock<Vec<ModelRecommendation>>,
    
    /// Server running flag
    running: RwLock<bool>,
}

impl WebInterface {
    /// Creates a new web interface
    pub fn new(
        config_manager: Arc<ConfigManager>,
        model_manager: Arc<ModelManager>,
        resource_manager: Arc<ResourceManager>,
        task_scheduler: Arc<TaskScheduler>,
        performance_optimizer: Arc<PerformanceOptimizer>,
        hardware_profiler: Arc<HardwareCapabilities>,
        search_engine: Arc<SearchEngine>,
        reasoning_engine: Arc<AdvancedReasoningEngine>,
    ) -> Result<Self> {
        // Get server address and port from config
        let address = config_manager
            .get_string("web_interface_address")
            .unwrap_or_else(|_| "0.0.0.0".to_string());
        
        let port = config_manager
            .get_u16("web_interface_port")
            .unwrap_or(80);
        
        Ok(Self {
            config_manager,
            model_manager,
            resource_manager,
            task_scheduler,
            performance_optimizer,
            hardware_profiler,
            search_engine,
            reasoning_engine,
            address,
            port,
            model_recommendations: RwLock::new(Vec::new()),
            running: RwLock::new(false),
        })
    }
    
    /// Starts the web interface
    pub async fn start(&self) -> Result<()> {
        // Check if already running
        {
            let running = self.running.read().await;
            if *running {
                return Ok(());
            }
        }
        
        // Set running flag
        {
            let mut running = self.running.write().await;
            *running = true;
        }
        
        info!("Starting web interface on {}:{}", self.address, self.port);
        
        // Generate model recommendations
        self.generate_model_recommendations().await?;
        
        // Start web server
        self.start_server().await?;
        
        // Open browser
        self.open_browser().await?;
        
        info!("Web interface started successfully");
        
        Ok(())
    }
    
    /// Stops the web interface
    pub async fn stop(&self) -> Result<()> {
        // Check if running
        {
            let running = self.running.read().await;
            if !*running {
                return Ok(());
            }
        }
        
        info!("Stopping web interface");
        
        // Set running flag to false
        {
            let mut running = self.running.write().await;
            *running = false;
        }
        
        info!("Web interface stopped successfully");
        
        Ok(())
    }
    
    /// Generates model recommendations based on hardware capabilities
    async fn generate_model_recommendations(&self) -> Result<()> {
        info!("Generating model recommendations");
        
        // Get hardware capabilities
        let cpu_info = self.hardware_profiler.get_cpu_info()?;
        let memory_info = self.hardware_profiler.get_memory_info()?;
        let gpu_info = self.hardware_profiler.get_gpu_info()?;
        
        // Get available models
        let available_models = self.model_manager.get_repository().get_all_models().await?;
        
        let mut recommendations = Vec::new();
        
        for model in available_models {
            // Calculate compatibility score
            let mut score = 100;
            
            // Check memory requirements
            let memory_ratio = (model.memory_requirements as f64) / (memory_info.total_bytes as f64);
            if memory_ratio > 0.8 {
                score -= 40;
            } else if memory_ratio > 0.5 {
                score -= 20;
            } else if memory_ratio > 0.3 {
                score -= 10;
            }
            
            // Check GPU requirements
            let requires_gpu = model.requires_gpu;
            let mut has_compatible_gpu = false;
            
            if requires_gpu {
                if gpu_info.is_empty() {
                    score -= 50;
                } else {
                    for gpu in &gpu_info {
                        if let Some(req_memory) = model.gpu_memory_requirements {
                            if gpu.memory_bytes >= req_memory {
                                has_compatible_gpu = true;
                                break;
                            }
                        } else {
                            has_compatible_gpu = true;
                            break;
                        }
                    }
                    
                    if !has_compatible_gpu {
                        score -= 40;
                    }
                }
            }
            
            // Determine if recommended
            let is_recommended = score >= 70;
            
            // Create recommendation
            let recommendation = ModelRecommendation {
                id: model.id.clone(),
                name: model.name.clone(),
                description: model.description.clone(),
                parameters: model.parameters,
                size_bytes: model.size_bytes,
                min_memory_bytes: model.memory_requirements,
                recommended_memory_bytes: (model.memory_requirements as f64 * 1.5) as u64,
                requires_gpu: model.requires_gpu,
                min_gpu_memory_bytes: model.gpu_memory_requirements,
                recommended_gpu_memory_bytes: model.gpu_memory_requirements.map(|mem| (mem as f64 * 1.5) as u64),
                compatibility_score: score as u8,
                is_recommended,
                tags: model.tags.clone(),
                capabilities: model.capabilities.clone(),
                download_url: model.download_url.clone(),
            };
            
            recommendations.push(recommendation);
        }
        
        // Sort by compatibility score (descending)
        recommendations.sort_by(|a, b| b.compatibility_score.cmp(&a.compatibility_score));
        
        // Update recommendations
        let mut model_recommendations = self.model_recommendations.write().await;
        *model_recommendations = recommendations;
        
        info!("Generated {} model recommendations", model_recommendations.len());
        
        Ok(())
    }
    
    /// Starts the web server
    async fn start_server(&self) -> Result<()> {
        let address = self.address.clone();
        let port = self.port;
        
        let model_manager = self.model_manager.clone();
        let resource_manager = self.resource_manager.clone();
        let task_scheduler = self.task_scheduler.clone();
        let performance_optimizer = self.performance_optimizer.clone();
        let hardware_profiler = self.hardware_profiler.clone();
        let search_engine = self.search_engine.clone();
        let reasoning_engine = self.reasoning_engine.clone();
        let model_recommendations = self.model_recommendations.clone();
        
        // Start server in a separate task
        tokio::spawn(async move {
            let model_manager_data = web::Data::new(model_manager);
            let resource_manager_data = web::Data::new(resource_manager);
            let task_scheduler_data = web::Data::new(task_scheduler);
            let performance_optimizer_data = web::Data::new(performance_optimizer);
            let hardware_profiler_data = web::Data::new(hardware_profiler);
            let search_engine_data = web::Data::new(search_engine);
            let reasoning_engine_data = web::Data::new(reasoning_engine);
            let model_recommendations_data = web::Data::new(model_recommendations);
            
            // Create and start HTTP server
            HttpServer::new(move || {
                App::new()
                    .wrap(Logger::default())
                    .wrap(Cors::permissive())
                    .app_data(model_manager_data.clone())
                    .app_data(resource_manager_data.clone())
                    .app_data(task_scheduler_data.clone())
                    .app_data(performance_optimizer_data.clone())
                    .app_data(hardware_profiler_data.clone())
                    .app_data(search_engine_data.clone())
                    .app_data(reasoning_engine_data.clone())
                    .app_data(model_recommendations_data.clone())
                    .service(
                        web::scope("/api")
                            .route("/status", web::get().to(get_system_status))
                            .route("/models/recommendations", web::get().to(get_model_recommendations))
                            .route("/models/download", web::post().to(download_model))
                            .route("/tasks", web::get().to(get_tasks))
                            .route("/tasks", web::post().to(submit_task))
                            .route("/tasks/{id}", web::get().to(get_task))
                            .route("/tasks/{id}/cancel", web::post().to(cancel_task))
                            .route("/search", web::post().to(search))
                            .route("/reasoning", web::post().to(submit_reasoning_task))
                            .route("/reasoning/{id}", web::get().to(get_reasoning_result))
                    )
                    .service(Files::new("/", "./web/dist").index_file("index.html"))
            })
            .bind(format!("{}:{}", address, port))
            .unwrap()
            .run()
            .await
            .unwrap();
        });
        
        // Wait a bit for server to start
        tokio::time::sleep(Duration::from_millis(500)).await;
        
        Ok(())
    }
    
    /// Opens the browser to the web interface
    async fn open_browser(&self) -> Result<()> {
        let url = format!("http://localhost:{}", self.port);
        
        info!("Opening browser to {}", url);
        
        // Detect platform and open browser
        #[cfg(target_os = "windows")]
        {
            Command::new("cmd")
                .args(&["/C", "start", &url])
                .spawn()
                .map_err(|e| Error::Internal(format!("Failed to open browser: {}", e)))?;
        }
        
        #[cfg(target_os = "macos")]
        {
            Command::new("open")
                .arg(&url)
                .spawn()
                .map_err(|e| Error::Internal(format!("Failed to open browser: {}", e)))?;
        }
        
        #[cfg(target_os = "linux")]
        {
            Command::new("xdg-open")
                .arg(&url)
                .spawn()
                .map_err(|e| Error::Internal(format!("Failed to open browser: {}", e)))?;
        }
        
        Ok(())
    }
}

/// Gets the system status
async fn get_system_status(
    resource_manager: web::Data<Arc<ResourceManager>>,
    task_scheduler: web::Data<Arc<TaskScheduler>>,
    model_manager: web::Data<Arc<ModelManager>>,
    hardware_profiler: web::Data<Arc<HardwareCapabilities>>,
) -> impl Responder {
    // Get CPU info
    let cpu_info = match hardware_profiler.get_cpu_info() {
        Ok(info) => CpuInfo {
            model: info.model,
            physical_cores: info.physical_cores,
            logical_cores: info.logical_cores,
            usage_percent: info.usage_percent,
        },
        Err(_) => CpuInfo {
            model: "Unknown".to_string(),
            physical_cores: 0,
            logical_cores: 0,
            usage_percent: 0.0,
        },
    };
    
    // Get memory info
    let memory_info = match hardware_profiler.get_memory_info() {
        Ok(info) => MemoryInfo {
            total_bytes: info.total_bytes,
            available_bytes: info.available_bytes,
            used_bytes: info.total_bytes - info.available_bytes,
            usage_percent: 100.0 * (1.0 - (info.available_bytes as f32 / info.total_bytes as f32)),
        },
        Err(_) => MemoryInfo {
            total_bytes: 0,
            available_bytes: 0,
            used_bytes: 0,
            usage_percent: 0.0,
        },
    };
    
    // Get GPU info
    let gpu_info = match hardware_profiler.get_gpu_info() {
        Ok(info) => {
            let mut gpus = Vec::new();
            
            for gpu in info {
                gpus.push(GpuInfo {
                    name: gpu.name,
                    total_memory_bytes: gpu.memory_bytes,
                    available_memory_bytes: gpu.available_memory_bytes,
                    usage_percent: gpu.usage_percent,
                });
            }
            
            gpus
        },
        Err(_) => Vec::new(),
    };
    
    // Get task info
    let running_tasks = task_scheduler.get_running_tasks_count();
    let queued_tasks = task_scheduler.get_queue_length().unwrap_or(0);
    
    let task_info = TaskInfo {
        running: running_tasks,
        queued: queued_tasks,
        completed: 0, // TODO: Get completed tasks count
    };
    
    // Get model info
    let model_info = ModelInfo {
        available: 0, // TODO: Get available models count
        loaded: 0,    // TODO: Get loaded models count
        total_size_bytes: 0, // TODO: Get total size of all models
    };
    
    // Create system status
    let status = SystemStatus {
        cpu: cpu_info,
        memory: memory_info,
        gpus: gpu_info,
        tasks: task_info,
        models: model_info,
    };
    
    HttpResponse::Ok().json(status)
}

/// Gets model recommendations
async fn get_model_recommendations(
    model_recommendations: web::Data<RwLock<Vec<ModelRecommendation>>>,
) -> impl Responder {
    let recommendations = model_recommendations.read().await.clone();
    HttpResponse::Ok().json(recommendations)
}

/// Downloads a model
async fn download_model(
    model_manager: web::Data<Arc<ModelManager>>,
    model_id: web::Json<String>,
) -> impl Responder {
    match model_manager.get_downloader().download_model(&model_id).await {
        Ok(_) => HttpResponse::Ok().json({"success": true}),
        Err(e) => HttpResponse::InternalServerError().json({"error": e.to_string()}),
    }
}

/// Gets all tasks
async fn get_tasks(
    task_scheduler: web::Data<Arc<TaskScheduler>>,
) -> impl Responder {
    match task_scheduler.get_all_tasks().await {
        Ok(tasks) => HttpResponse::Ok().json(tasks),
        Err(e) => HttpResponse::InternalServerError().json({"error": e.to_string()}),
    }
}

/// Submits a task
async fn submit_task(
    task_scheduler: web::Data<Arc<TaskScheduler>>,
    task_definition: web::Json<task_scheduler::TaskDefinition>,
) -> impl Responder {
    match task_scheduler.submit_task(task_definition.into_inner()).await {
        Ok(task_id) => HttpResponse::Ok().json({"task_id": task_id}),
        Err(e) => HttpResponse::InternalServerError().json({"error": e.to_string()}),
    }
}

/// Gets a task
async fn get_task(
    task_scheduler: web::Data<Arc<TaskScheduler>>,
    path: web::Path<String>,
) -> impl Responder {
    let task_id = path.into_inner();
    
    // Get task status
    let status = match task_scheduler.get_task_status(&task_id).await {
        Ok(status) => status,
        Err(e) => return HttpResponse::NotFound().json({"error": e.to_string()}),
    };
    
    // If task is completed, get result
    if status == TaskStatus::Completed {
        match task_scheduler.get_task_result(&task_id) {
            Ok(result) => HttpResponse::Ok().json(result),
            Err(e) => HttpResponse::InternalServerError().json({"error": e.to_string()}),
        }
    } else {
        // Return status only
        HttpResponse::Ok().json({"task_id": task_id, "status": status})
    }
}

/// Cancels a task
async fn cancel_task(
    task_scheduler: web::Data<Arc<TaskScheduler>>,
    path: web::Path<String>,
) -> impl Responder {
    let task_id = path.into_inner();
    
    match task_scheduler.cancel_task(&task_id).await {
        Ok(_) => HttpResponse::Ok().json({"success": true}),
        Err(e) => HttpResponse::InternalServerError().json({"error": e.to_string()}),
    }
}

/// Performs a search
async fn search(
    search_engine: web::Data<Arc<SearchEngine>>,
    query: web::Json<search_engine::SearchQuery>,
) -> impl Responder {
    match search_engine.search(query.into_inner()).await {
        Ok(results) => HttpResponse::Ok().json(results),
        Err(e) => HttpResponse::InternalServerError().json({"error": e.to_string()}),
    }
}

/// Submits a reasoning task
async fn submit_reasoning_task(
    reasoning_engine: web::Data<Arc<AdvancedReasoningEngine>>,
    task: web::Json<advanced_reasoning::ReasoningTask>,
) -> impl Responder {
    match reasoning_engine.submit_task(task.into_inner()).await {
        Ok(task_id) => HttpResponse::Ok().json({"task_id": task_id}),
        Err(e) => HttpResponse::InternalServerError().json({"error": e.to_string()}),
    }
}

/// Gets a reasoning result
async fn get_reasoning_result(
    reasoning_engine: web::Data<Arc<AdvancedReasoningEngine>>,
    path: web::Path<String>,
) -> impl Responder {
    let task_id = path.into_inner();
    
    match reasoning_engine.get_result(&task_id).await {
        Ok(result) => HttpResponse::Ok().json(result),
        Err(e) => {
            // Check if task is still active
            if reasoning_engine.is_task_active(&task_id).await {
                HttpResponse::Ok().json({"task_id": task_id, "status": "running"})
            } else {
                HttpResponse::NotFound().json({"error": e.to_string()})
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    // These tests would require mocking the dependencies
    // For now, we just have placeholder tests
    
    #[test]
    fn test_web_interface() {
        // Placeholder test
        assert!(true);
    }
}

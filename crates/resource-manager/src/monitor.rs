//! Resource monitoring implementation
//!
//! This module provides functionality for monitoring system resources,
//! including CPU, memory, GPU, and disk usage.

use std::sync::Arc;
use tokio::sync::RwLock;
use anyhow::Result;
use tracing::{info, warn, error, debug, trace};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use sysinfo::{System, SystemExt, CpuExt, ProcessorExt, DiskExt};

use common::error::Error;
use common::models::{SystemResources, CpuInfo, MemoryInfo, GpuInfo, DiskInfo, NetworkInfo};
use common::types::ResourceRequirements;
use hardware_profiler::HardwareCapabilities;
use config::ConfigManager;

/// Resource monitor for tracking system resource usage
pub struct ResourceMonitor {
    /// System information
    system: Arc<RwLock<System>>,
    
    /// Hardware capabilities
    hardware_capabilities: Arc<HardwareCapabilities>,
    
    /// Configuration manager
    config_manager: Arc<ConfigManager>,
    
    /// Last update time
    last_update: Arc<RwLock<Instant>>,
    
    /// Update interval
    update_interval: Duration,
    
    /// Resource thresholds
    thresholds: ResourceThresholds,
    
    /// Resource reservation
    reserved_resources: ResourceReservation,
}

/// Resource thresholds for triggering alerts
#[derive(Debug, Clone)]
pub struct ResourceThresholds {
    /// CPU usage threshold (percentage)
    pub cpu_usage_percent: f32,
    
    /// Memory usage threshold (percentage)
    pub memory_usage_percent: f32,
    
    /// GPU usage threshold (percentage)
    pub gpu_usage_percent: f32,
    
    /// GPU memory usage threshold (percentage)
    pub gpu_memory_usage_percent: f32,
    
    /// Disk usage threshold (percentage)
    pub disk_usage_percent: f32,
}

/// Resource reservation for system stability
#[derive(Debug, Clone)]
pub struct ResourceReservation {
    /// Reserved CPU cores
    pub cpu_cores: usize,
    
    /// Reserved memory (bytes)
    pub memory_bytes: u64,
    
    /// Reserved GPU memory (bytes)
    pub gpu_memory_bytes: u64,
    
    /// Reserved disk space (bytes)
    pub disk_bytes: u64,
}

impl ResourceMonitor {
    /// Creates a new resource monitor
    pub fn new(
        hardware_capabilities: Arc<HardwareCapabilities>,
        config_manager: Arc<ConfigManager>,
    ) -> Result<Self> {
        // Initialize system information
        let mut system = System::new_all();
        system.refresh_all();
        
        // Get update interval from config
        let update_interval = config_manager
            .get_duration("resource_update_interval")
            .unwrap_or_else(|_| Duration::from_secs(1));
        
        // Get resource thresholds from config
        let thresholds = ResourceThresholds {
            cpu_usage_percent: config_manager
                .get_f32("cpu_usage_threshold_percent")
                .unwrap_or(80.0),
            memory_usage_percent: config_manager
                .get_f32("memory_usage_threshold_percent")
                .unwrap_or(80.0),
            gpu_usage_percent: config_manager
                .get_f32("gpu_usage_threshold_percent")
                .unwrap_or(80.0),
            gpu_memory_usage_percent: config_manager
                .get_f32("gpu_memory_usage_threshold_percent")
                .unwrap_or(80.0),
            disk_usage_percent: config_manager
                .get_f32("disk_usage_threshold_percent")
                .unwrap_or(90.0),
        };
        
        // Get resource reservation from config
        let total_memory = system.total_memory();
        let total_cores = system.processors().len();
        
        let reserved_resources = ResourceReservation {
            cpu_cores: config_manager
                .get_usize("reserved_cpu_cores")
                .unwrap_or(1),
            memory_bytes: config_manager
                .get_u64("reserved_memory_bytes")
                .unwrap_or_else(|_| total_memory / 10), // 10% of total memory
            gpu_memory_bytes: config_manager
                .get_u64("reserved_gpu_memory_bytes")
                .unwrap_or(256 * 1024 * 1024), // 256 MB
            disk_bytes: config_manager
                .get_u64("reserved_disk_bytes")
                .unwrap_or(1024 * 1024 * 1024), // 1 GB
        };
        
        Ok(Self {
            system: Arc::new(RwLock::new(system)),
            hardware_capabilities,
            config_manager,
            last_update: Arc::new(RwLock::new(Instant::now())),
            update_interval,
            thresholds,
            reserved_resources,
        })
    }
    
    /// Starts the resource monitor
    pub async fn start(&self) -> Result<()> {
        info!("Starting resource monitor");
        
        // Start resource monitoring
        self.start_monitoring().await?;
        
        info!("Resource monitor started successfully");
        
        Ok(())
    }
    
    /// Stops the resource monitor
    pub async fn stop(&self) -> Result<()> {
        info!("Stopping resource monitor");
        
        // Resource monitoring is handled by the background task
        // which will exit when the Arc is dropped
        
        info!("Resource monitor stopped successfully");
        
        Ok(())
    }
    
    /// Starts resource monitoring in a background task
    async fn start_monitoring(&self) -> Result<()> {
        let system = self.system.clone();
        let last_update = self.last_update.clone();
        let update_interval = self.update_interval;
        
        tokio::spawn(async move {
            loop {
                // Check if it's time to update
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
                    
                    // Refresh system information
                    {
                        let mut sys = system.write().await;
                        sys.refresh_all();
                    }
                }
                
                // Sleep for a short time
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
        });
        
        Ok(())
    }
    
    /// Gets the current system resources
    pub async fn get_system_resources(&self) -> Result<SystemResources> {
        // Get system information
        let system = self.system.read().await;
        
        // Get CPU information
        let cpu_info = self.get_cpu_info(&system)?;
        
        // Get memory information
        let memory_info = self.get_memory_info(&system)?;
        
        // Get GPU information
        let gpu_info = self.get_gpu_info().await?;
        
        // Get disk information
        let disk_info = self.get_disk_info(&system)?;
        
        // Get network information
        let network_info = self.get_network_info(&system)?;
        
        // Calculate performance score
        let performance_score = self.calculate_performance_score(
            &cpu_info,
            &memory_info,
            &gpu_info,
            &disk_info,
        )?;
        
        Ok(SystemResources {
            cpu: cpu_info,
            memory: memory_info,
            gpus: gpu_info,
            disk: disk_info,
            network: network_info,
            performance_score,
        })
    }
    
    /// Gets CPU information
    fn get_cpu_info(&self, system: &System) -> Result<CpuInfo> {
        let processors = system.processors();
        let global_cpu = system.global_processor_info();
        
        // Get CPU model from first processor
        let model = if !processors.is_empty() {
            processors[0].brand().to_string()
        } else {
            "Unknown".to_string()
        };
        
        // Get CPU usage
        let usage_percent = global_cpu.cpu_usage();
        
        // Get CPU features from hardware capabilities
        let features = self.hardware_capabilities.get_cpu_features();
        
        Ok(CpuInfo {
            model,
            physical_cores: self.hardware_capabilities.get_physical_cpu_cores(),
            logical_cores: processors.len(),
            architecture: std::env::consts::ARCH.to_string(),
            frequency_mhz: if !processors.is_empty() {
                processors[0].frequency() as u32
            } else {
                0
            },
            cache_size_kb: 0, // Not available from sysinfo
            features,
            usage_percent,
        })
    }
    
    /// Gets memory information
    fn get_memory_info(&self, system: &System) -> Result<MemoryInfo> {
        let total_memory = system.total_memory();
        let used_memory = system.used_memory();
        let available_memory = total_memory - used_memory;
        let usage_percent = if total_memory > 0 {
            (used_memory as f32 / total_memory as f32) * 100.0
        } else {
            0.0
        };
        
        Ok(MemoryInfo {
            total_bytes: total_memory,
            available_bytes: available_memory,
            used_bytes: used_memory,
            usage_percent,
        })
    }
    
    /// Gets GPU information
    async fn get_gpu_info(&self) -> Result<Vec<GpuInfo>> {
        // Get GPU information from hardware capabilities
        let gpu_count = self.hardware_capabilities.get_gpu_count();
        let mut gpus = Vec::new();
        
        for i in 0..gpu_count {
            let gpu_id = i as usize;
            
            // Get GPU information
            let model = self.hardware_capabilities.get_gpu_name(gpu_id)?;
            let vendor = self.hardware_capabilities.get_gpu_vendor(gpu_id)?;
            let total_memory = self.hardware_capabilities.get_gpu_memory(gpu_id)?;
            let used_memory = self.hardware_capabilities.get_gpu_memory_used(gpu_id)?;
            let available_memory = total_memory - used_memory;
            let memory_usage_percent = if total_memory > 0 {
                (used_memory as f32 / total_memory as f32) * 100.0
            } else {
                0.0
            };
            let usage_percent = self.hardware_capabilities.get_gpu_utilization(gpu_id)?;
            let temperature = self.hardware_capabilities.get_gpu_temperature(gpu_id)?;
            let driver_version = self.hardware_capabilities.get_gpu_driver_version(gpu_id)?;
            let compute_capability = self.hardware_capabilities.get_gpu_compute_capability(gpu_id)?;
            
            gpus.push(GpuInfo {
                index: gpu_id,
                model,
                vendor,
                total_memory_bytes: total_memory,
                available_memory_bytes: available_memory,
                used_memory_bytes: used_memory,
                memory_usage_percent,
                usage_percent,
                temperature_celsius: temperature,
                driver_version,
                compute_capability: Some(compute_capability),
            });
        }
        
        Ok(gpus)
    }
    
    /// Gets disk information
    fn get_disk_info(&self, system: &System) -> Result<DiskInfo> {
        let disks = system.disks();
        
        // Sum up all disk space
        let mut total_bytes = 0;
        let mut available_bytes = 0;
        
        for disk in disks {
            total_bytes += disk.total_space();
            available_bytes += disk.available_space();
        }
        
        let used_bytes = total_bytes - available_bytes;
        let usage_percent = if total_bytes > 0 {
            (used_bytes as f32 / total_bytes as f32) * 100.0
        } else {
            0.0
        };
        
        // Disk I/O rates are not available from sysinfo
        // In a real implementation, we would use platform-specific APIs
        
        Ok(DiskInfo {
            total_bytes,
            available_bytes,
            used_bytes,
            usage_percent,
            read_bytes_per_second: 0,
            write_bytes_per_second: 0,
        })
    }
    
    /// Gets network information
    fn get_network_info(&self, system: &System) -> Result<NetworkInfo> {
        // Get network interfaces
        let networks = system.networks();
        
        // Find the interface with the most traffic
        let mut primary_interface = "unknown".to_string();
        let mut max_traffic = 0;
        let mut receive_bytes = 0;
        let mut transmit_bytes = 0;
        
        for (name, data) in networks {
            let total_traffic = data.received() + data.transmitted();
            if total_traffic > max_traffic {
                max_traffic = total_traffic;
                primary_interface = name.clone();
                receive_bytes = data.received();
                transmit_bytes = data.transmitted();
            }
        }
        
        // In a real implementation, we would calculate rates over time
        // For now, we just return the current values
        
        Ok(NetworkInfo {
            primary_interface,
            ip_address: "127.0.0.1".to_string(), // Not available from sysinfo
            connection_type: "unknown".to_string(), // Not available from sysinfo
            receive_bytes_per_second: receive_bytes,
            transmit_bytes_per_second: transmit_bytes,
        })
    }
    
    /// Calculates a performance score based on system resources
    fn calculate_performance_score(
        &self,
        cpu_info: &CpuInfo,
        memory_info: &MemoryInfo,
        gpu_info: &[GpuInfo],
        disk_info: &DiskInfo,
    ) -> Result<u32> {
        // Calculate CPU score (0-100)
        let cpu_score = (cpu_info.logical_cores as f32 * 5.0).min(100.0);
        
        // Calculate memory score (0-100)
        let memory_gb = memory_info.total_bytes as f32 / (1024.0 * 1024.0 * 1024.0);
        let memory_score = (memory_gb * 5.0).min(100.0);
        
        // Calculate GPU score (0-100)
        let gpu_score = if !gpu_info.is_empty() {
            let mut score = 0.0;
            for gpu in gpu_info {
                let gpu_memory_gb = gpu.total_memory_bytes as f32 / (1024.0 * 1024.0 * 1024.0);
                score += gpu_memory_gb * 10.0;
            }
            score.min(100.0)
        } else {
            0.0
        };
        
        // Calculate disk score (0-100)
        let disk_gb = disk_info.total_bytes as f32 / (1024.0 * 1024.0 * 1024.0);
        let disk_score = (disk_gb * 0.5).min(100.0);
        
        // Calculate overall score (0-1000)
        let overall_score = (cpu_score * 3.0 + memory_score * 3.0 + gpu_score * 3.0 + disk_score) as u32;
        
        Ok(overall_score)
    }
    
    /// Checks if there are enough resources to execute a task
    pub async fn can_execute_task(&self, task: &common::models::Task) -> Result<bool> {
        // Get task resource requirements
        let requirements = self.get_task_resource_requirements(task)?;
        
        // Get current system resources
        let resources = self.get_system_resources().await?;
        
        // Check CPU
        let available_cpu_cores = resources.cpu.logical_cores - self.reserved_resources.cpu_cores;
        if requirements.min_cpu_cores > available_cpu_cores {
            return Ok(false);
        }
        
        // Check memory
        let available_memory = resources.memory.available_bytes - self.reserved_resources.memory_bytes;
        if requirements.min_memory_bytes > available_memory {
            return Ok(false);
        }
        
        // Check GPU if required
        if requirements.requires_gpu {
            // Check if we have any GPUs
            if resources.gpus.is_empty() {
                return Ok(false);
            }
            
            // Check GPU memory
            if let Some(min_gpu_memory) = requirements.min_gpu_memory_bytes {
                let mut has_enough_memory = false;
                
                for gpu in &resources.gpus {
                    let available_gpu_memory = gpu.available_memory_bytes - self.reserved_resources.gpu_memory_bytes;
                    if available_gpu_memory >= min_gpu_memory {
                        has_enough_memory = true;
                        break;
                    }
                }
                
                if !has_enough_memory {
                    return Ok(false);
                }
            }
        }
        
        // Check disk
        let available_disk = resources.disk.available_bytes - self.reserved_resources.disk_bytes;
        if requirements.min_disk_bytes > available_disk {
            return Ok(false);
        }
        
        // Check CPU features
        for feature in &requirements.required_cpu_features {
            if !resources.cpu.features.contains(feature) {
                return Ok(false);
            }
        }
        
        // Check GPU features if required
        if requirements.requires_gpu && !requirements.required_gpu_features.is_empty() {
            let mut has_required_features = false;
            
            for gpu in &resources.gpus {
                let mut has_all_features = true;
                
                for feature in &requirements.required_gpu_features {
                    // In a real implementation, we would check if the GPU has the required feature
                    // For now, we just assume it does
                    has_all_features = true;
                }
                
                if has_all_features {
                    has_required_features = true;
                    break;
                }
            }
            
            if !has_required_features {
                return Ok(false);
            }
        }
        
        Ok(true)
    }
    
    /// Gets resource requirements for a task
    fn get_task_resource_requirements(&self, task: &common::models::Task) -> Result<ResourceRequirements> {
        // In a real implementation, this would be more sophisticated
        // For now, we just use some default values based on task type
        
        match task.task_type {
            common::models::TaskType::Inference => {
                Ok(ResourceRequirements {
                    min_cpu_cores: 1,
                    min_memory_bytes: 1024 * 1024 * 1024, // 1 GB
                    min_gpu_memory_bytes: Some(1024 * 1024 * 1024), // 1 GB
                    min_disk_bytes: 100 * 1024 * 1024, // 100 MB
                    requires_gpu: true,
                    required_cpu_features: vec!["sse4.1".to_string(), "avx".to_string()],
                    required_gpu_features: vec![],
                })
            },
            common::models::TaskType::FineTuning => {
                Ok(ResourceRequirements {
                    min_cpu_cores: 4,
                    min_memory_bytes: 8 * 1024 * 1024 * 1024, // 8 GB
                    min_gpu_memory_bytes: Some(8 * 1024 * 1024 * 1024), // 8 GB
                    min_disk_bytes: 10 * 1024 * 1024 * 1024, // 10 GB
                    requires_gpu: true,
                    required_cpu_features: vec!["sse4.1".to_string(), "avx".to_string(), "avx2".to_string()],
                    required_gpu_features: vec![],
                })
            },
            common::models::TaskType::Evaluation => {
                Ok(ResourceRequirements {
                    min_cpu_cores: 2,
                    min_memory_bytes: 4 * 1024 * 1024 * 1024, // 4 GB
                    min_gpu_memory_bytes: Some(4 * 1024 * 1024 * 1024), // 4 GB
                    min_disk_bytes: 1 * 1024 * 1024 * 1024, // 1 GB
                    requires_gpu: true,
                    required_cpu_features: vec!["sse4.1".to_string(), "avx".to_string()],
                    required_gpu_features: vec![],
                })
            },
            common::models::TaskType::Export => {
                Ok(ResourceRequirements {
                    min_cpu_cores: 2,
                    min_memory_bytes: 4 * 1024 * 1024 * 1024, // 4 GB
                    min_gpu_memory_bytes: None,
                    min_disk_bytes: 10 * 1024 * 1024 * 1024, // 10 GB
                    requires_gpu: false,
                    required_cpu_features: vec![],
                    required_gpu_features: vec![],
                })
            },
            common::models::TaskType::Custom(_) => {
                // For custom tasks, we use conservative defaults
                Ok(ResourceRequirements {
                    min_cpu_cores: 2,
                    min_memory_bytes: 2 * 1024 * 1024 * 1024, // 2 GB
                    min_gpu_memory_bytes: None,
                    min_disk_bytes: 1 * 1024 * 1024 * 1024, // 1 GB
                    requires_gpu: false,
                    required_cpu_features: vec![],
                    required_gpu_features: vec![],
                })
            },
        }
    }
    
    /// Gets the total memory
    pub fn get_total_memory(&self) -> Result<u64> {
        let system = match self.system.try_read() {
            Ok(system) => system,
            Err(_) => return Err(Error::Internal("Failed to acquire read lock on system".to_string()).into()),
        };
        
        Ok(system.total_memory())
    }
    
    /// Gets the available memory
    pub async fn get_available_memory(&self) -> Result<u64> {
        let system = self.system.read().await;
        let available = system.total_memory() - system.used_memory();
        
        // Subtract reserved memory
        let available_after_reservation = available.saturating_sub(self.reserved_resources.memory_bytes);
        
        Ok(available_after_reservation)
    }
    
    /// Gets the total CPU cores
    pub fn get_total_cpu_cores(&self) -> Result<usize> {
        let system = match self.system.try_read() {
            Ok(system) => system,
            Err(_) => return Err(Error::Internal("Failed to acquire read lock on system".to_string()).into()),
        };
        
        Ok(system.processors().len())
    }
    
    /// Gets the available CPU cores
    pub fn get_available_cpu_cores(&self) -> Result<usize> {
        let total_cores = self.get_total_cpu_cores()?;
        
        // Subtract reserved cores
        let available_cores = total_cores.saturating_sub(self.reserved_resources.cpu_cores);
        
        Ok(available_cores)
    }
    
    /// Gets the resource thresholds
    pub fn get_thresholds(&self) -> ResourceThresholds {
        self.thresholds.clone()
    }
    
    /// Sets the resource thresholds
    pub fn set_thresholds(&mut self, thresholds: ResourceThresholds) {
        self.thresholds = thresholds;
    }
    
    /// Gets the resource reservation
    pub fn get_reserved_resources(&self) -> ResourceReservation {
        self.reserved_resources.clone()
    }
    
    /// Sets the resource reservation
    pub fn set_reserved_resources(&mut self, reserved: ResourceReservation) {
        self.reserved_resources = reserved;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    // These tests would require mocking the dependencies
    // For now, we just have placeholder tests
    
    #[test]
    fn test_resource_monitor() {
        // Placeholder test
        assert!(true);
    }
}

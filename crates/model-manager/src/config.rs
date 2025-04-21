//! Model configuration management
//!
//! This module provides functionality for managing model configurations,
//! including optimal settings for different hardware and model types.

use std::sync::Arc;
use anyhow::Result;
use tracing::{info, warn, error, debug};
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

use common::error::Error;
use common::models::{Model, ModelConfig, Quantization, Device};
use hardware_profiler::{HardwareCapabilities, SystemProfiler};
use config::ConfigManager;

/// Model configuration manager
pub struct ModelConfigManager {
    /// Hardware capabilities
    hardware_capabilities: Arc<HardwareCapabilities>,
    
    /// System profiler
    system_profiler: Arc<SystemProfiler>,
    
    /// Configuration manager
    config_manager: Arc<ConfigManager>,
    
    /// Default configurations by model type
    default_configs: HashMap<String, ModelConfig>,
}

impl ModelConfigManager {
    /// Creates a new model configuration manager
    pub fn new(
        hardware_capabilities: Arc<HardwareCapabilities>,
        system_profiler: Arc<SystemProfiler>,
        config_manager: Arc<ConfigManager>,
    ) -> Result<Self> {
        let mut manager = Self {
            hardware_capabilities,
            system_profiler,
            config_manager,
            default_configs: HashMap::new(),
        };
        
        // Initialize default configurations
        manager.initialize_default_configs()?;
        
        Ok(manager)
    }
    
    /// Initializes default configurations for different model types
    fn initialize_default_configs(&mut self) -> Result<()> {
        // Get hardware info
        let has_gpu = self.hardware_capabilities.has_gpu();
        let has_avx512 = self.hardware_capabilities.has_cpu_feature("avx512f");
        let has_avx2 = self.hardware_capabilities.has_cpu_feature("avx2");
        let total_memory = self.hardware_capabilities.get_total_memory_bytes();
        
        // Default config for small models (1B-3B parameters)
        let small_config = ModelConfig {
            quantization: if has_gpu {
                Quantization::FP16
            } else if has_avx512 || has_avx2 {
                Quantization::INT8
            } else {
                Quantization::INT4
            },
            device: if has_gpu {
                Device::CUDA(0)
            } else {
                Device::CPU
            },
            context_size: 4096,
            tensor_parallel: false,
            pipeline_parallel: false,
            tensor_parallel_shards: 1,
            pipeline_parallel_shards: 1,
            flash_attention: has_gpu,
            continuous_batching: true,
            custom_options: HashMap::new(),
        };
        
        // Default config for medium models (7B-13B parameters)
        let medium_config = ModelConfig {
            quantization: if has_gpu && total_memory > 16 * 1024 * 1024 * 1024 {
                Quantization::FP16
            } else if has_avx512 {
                Quantization::INT8
            } else {
                Quantization::INT4
            },
            device: if has_gpu {
                Device::CUDA(0)
            } else {
                Device::CPU
            },
            context_size: 4096,
            tensor_parallel: has_gpu && self.hardware_capabilities.get_gpu_count() > 1,
            pipeline_parallel: false,
            tensor_parallel_shards: if has_gpu && self.hardware_capabilities.get_gpu_count() > 1 {
                self.hardware_capabilities.get_gpu_count() as usize
            } else {
                1
            },
            pipeline_parallel_shards: 1,
            flash_attention: has_gpu,
            continuous_batching: true,
            custom_options: HashMap::new(),
        };
        
        // Default config for large models (30B-70B parameters)
        let large_config = ModelConfig {
            quantization: if has_gpu && total_memory > 32 * 1024 * 1024 * 1024 {
                Quantization::FP16
            } else if has_avx512 {
                Quantization::INT8
            } else {
                Quantization::INT4
            },
            device: if has_gpu {
                if self.hardware_capabilities.get_gpu_count() > 1 {
                    Device::MultiGPU((0..self.hardware_capabilities.get_gpu_count() as usize).collect())
                } else {
                    Device::CUDA(0)
                }
            } else {
                Device::CPU
            },
            context_size: 4096,
            tensor_parallel: has_gpu && self.hardware_capabilities.get_gpu_count() > 1,
            pipeline_parallel: has_gpu && self.hardware_capabilities.get_gpu_count() > 2,
            tensor_parallel_shards: if has_gpu && self.hardware_capabilities.get_gpu_count() > 1 {
                self.hardware_capabilities.get_gpu_count() as usize
            } else {
                1
            },
            pipeline_parallel_shards: if has_gpu && self.hardware_capabilities.get_gpu_count() > 2 {
                2
            } else {
                1
            },
            flash_attention: has_gpu,
            continuous_batching: true,
            custom_options: HashMap::new(),
        };
        
        // Default config for huge models (100B+ parameters)
        let huge_config = ModelConfig {
            quantization: if has_gpu && total_memory > 64 * 1024 * 1024 * 1024 {
                Quantization::FP16
            } else {
                Quantization::INT4
            },
            device: if has_gpu {
                if self.hardware_capabilities.get_gpu_count() > 1 {
                    Device::MultiGPU((0..self.hardware_capabilities.get_gpu_count() as usize).collect())
                } else {
                    Device::CUDA(0)
                }
            } else {
                Device::CPU
            },
            context_size: 4096,
            tensor_parallel: has_gpu,
            pipeline_parallel: has_gpu && self.hardware_capabilities.get_gpu_count() > 1,
            tensor_parallel_shards: if has_gpu {
                self.hardware_capabilities.get_gpu_count() as usize
            } else {
                1
            },
            pipeline_parallel_shards: if has_gpu && self.hardware_capabilities.get_gpu_count() > 1 {
                self.hardware_capabilities.get_gpu_count() as usize / 2
            } else {
                1
            },
            flash_attention: has_gpu,
            continuous_batching: true,
            custom_options: HashMap::new(),
        };
        
        // Add default configs
        self.default_configs.insert("small".to_string(), small_config);
        self.default_configs.insert("medium".to_string(), medium_config);
        self.default_configs.insert("large".to_string(), large_config);
        self.default_configs.insert("huge".to_string(), huge_config);
        
        Ok(())
    }
    
    /// Gets the optimal configuration for a model
    pub fn get_optimal_config(&self, model: &Model) -> Result<ModelConfig> {
        // Determine model size category
        let size_category = self.get_model_size_category(model.size);
        
        // Get default config for this size category
        let default_config = self.default_configs.get(&size_category)
            .ok_or_else(|| Error::Internal(format!("No default config for size category: {}", size_category)))?;
        
        // Start with the default config
        let mut config = default_config.clone();
        
        // Adjust based on available resources
        self.adjust_config_for_resources(&mut config, model)?;
        
        // Adjust based on model-specific requirements
        self.adjust_config_for_model(&mut config, model)?;
        
        // Apply any user overrides from config
        self.apply_user_overrides(&mut config, model)?;
        
        Ok(config)
    }
    
    /// Gets the size category for a model based on its parameter count
    fn get_model_size_category(&self, size_bytes: u64) -> String {
        // Estimate parameter count (assuming FP32 parameters)
        let param_count = size_bytes / 4;
        
        if param_count < 3_000_000_000 {
            "small".to_string()
        } else if param_count < 15_000_000_000 {
            "medium".to_string()
        } else if param_count < 80_000_000_000 {
            "large".to_string()
        } else {
            "huge".to_string()
        }
    }
    
    /// Adjusts configuration based on available resources
    fn adjust_config_for_resources(&self, config: &mut ModelConfig, model: &Model) -> Result<()> {
        // Get available memory
        let available_memory = self.system_profiler.get_available_memory_bytes()?;
        
        // Estimate model memory requirements
        let memory_required = self.estimate_memory_requirements(model, config)?;
        
        // If we don't have enough memory, adjust quantization
        if memory_required > available_memory {
            debug!(
                "Adjusting quantization due to memory constraints: required={}, available={}",
                memory_required, available_memory
            );
            
            // Try progressively more aggressive quantization
            if matches!(config.quantization, Quantization::None) {
                config.quantization = Quantization::FP16;
            } else if matches!(config.quantization, Quantization::FP16) || matches!(config.quantization, Quantization::BF16) {
                config.quantization = Quantization::INT8;
            } else if matches!(config.quantization, Quantization::INT8) {
                config.quantization = Quantization::INT4;
            }
            
            // Recalculate memory requirements
            let new_memory_required = self.estimate_memory_requirements(model, config)?;
            
            // If still not enough, disable tensor parallelism
            if new_memory_required > available_memory && config.tensor_parallel {
                config.tensor_parallel = false;
                config.tensor_parallel_shards = 1;
            }
        }
        
        Ok(())
    }
    
    /// Adjusts configuration based on model-specific requirements
    fn adjust_config_for_model(&self, config: &mut ModelConfig, model: &Model) -> Result<()> {
        // Check model metadata for specific requirements
        if let Some(context_size) = model.metadata.get("context_size") {
            if let Ok(size) = context_size.parse::<usize>() {
                config.context_size = size;
            }
        }
        
        // Check if model supports flash attention
        if let Some(supports_flash) = model.metadata.get("supports_flash_attention") {
            if supports_flash == "false" {
                config.flash_attention = false;
            }
        }
        
        // Check if model requires specific quantization
        if let Some(required_quantization) = model.metadata.get("required_quantization") {
            match required_quantization.as_str() {
                "fp32" => config.quantization = Quantization::None,
                "fp16" => config.quantization = Quantization::FP16,
                "bf16" => config.quantization = Quantization::BF16,
                "int8" => config.quantization = Quantization::INT8,
                "int4" => config.quantization = Quantization::INT4,
                custom => config.quantization = Quantization::Custom(custom.to_string()),
            }
        }
        
        Ok(())
    }
    
    /// Applies user overrides from configuration
    fn apply_user_overrides(&self, config: &mut ModelConfig, model: &Model) -> Result<()> {
        // Check for user overrides in config
        let prefix = format!("model.{}.config.", model.id);
        
        // Check for quantization override
        if let Ok(quantization) = self.config_manager.get_string(&format!("{}quantization", prefix)) {
            match quantization.as_str() {
                "fp32" => config.quantization = Quantization::None,
                "fp16" => config.quantization = Quantization::FP16,
                "bf16" => config.quantization = Quantization::BF16,
                "int8" => config.quantization = Quantization::INT8,
                "int4" => config.quantization = Quantization::INT4,
                custom => config.quantization = Quantization::Custom(custom.to_string()),
            }
        }
        
        // Check for device override
        if let Ok(device) = self.config_manager.get_string(&format!("{}device", prefix)) {
            if device == "cpu" {
                config.device = Device::CPU;
            } else if device.starts_with("cuda:") {
                if let Ok(gpu_id) = device[5..].parse::<usize>() {
                    config.device = Device::CUDA(gpu_id);
                }
            } else if device.starts_with("rocm:") {
                if let Ok(gpu_id) = device[5..].parse::<usize>() {
                    config.device = Device::ROCm(gpu_id);
                }
            } else if device == "multi-gpu" {
                let gpu_count = self.hardware_capabilities.get_gpu_count() as usize;
                if gpu_count > 1 {
                    config.device = Device::MultiGPU((0..gpu_count).collect());
                }
            }
        }
        
        // Check for context size override
        if let Ok(context_size) = self.config_manager.get_usize(&format!("{}context_size", prefix)) {
            config.context_size = context_size;
        }
        
        // Check for tensor parallelism override
        if let Ok(tensor_parallel) = self.config_manager.get_bool(&format!("{}tensor_parallel", prefix)) {
            config.tensor_parallel = tensor_parallel;
        }
        
        // Check for pipeline parallelism override
        if let Ok(pipeline_parallel) = self.config_manager.get_bool(&format!("{}pipeline_parallel", prefix)) {
            config.pipeline_parallel = pipeline_parallel;
        }
        
        // Check for tensor parallel shards override
        if let Ok(shards) = self.config_manager.get_usize(&format!("{}tensor_parallel_shards", prefix)) {
            config.tensor_parallel_shards = shards;
        }
        
        // Check for pipeline parallel shards override
        if let Ok(shards) = self.config_manager.get_usize(&format!("{}pipeline_parallel_shards", prefix)) {
            config.pipeline_parallel_shards = shards;
        }
        
        // Check for flash attention override
        if let Ok(flash) = self.config_manager.get_bool(&format!("{}flash_attention", prefix)) {
            config.flash_attention = flash;
        }
        
        // Check for continuous batching override
        if let Ok(batching) = self.config_manager.get_bool(&format!("{}continuous_batching", prefix)) {
            config.continuous_batching = batching;
        }
        
        Ok(())
    }
    
    /// Estimates memory requirements for a model with a given configuration
    fn estimate_memory_requirements(&self, model: &Model, config: &ModelConfig) -> Result<u64> {
        // Base memory requirement is the model size
        let base_size = model.size;
        
        // Adjust based on quantization
        let size_multiplier = match config.quantization {
            Quantization::None => 4.0,     // FP32
            Quantization::FP16 => 2.0,     // FP16
            Quantization::BF16 => 2.0,     // BF16
            Quantization::INT8 => 1.0,     // INT8
            Quantization::INT4 => 0.5,     // INT4
            Quantization::Custom(_) => 1.0, // Assume INT8 for custom
        };
        
        // Calculate memory required for model weights
        let weights_memory = (base_size as f64 * size_multiplier) as u64;
        
        // Add memory for KV cache based on context size
        let kv_cache_per_token = match config.quantization {
            Quantization::None => 8,     // FP32
            Quantization::FP16 => 4,     // FP16
            Quantization::BF16 => 4,     // BF16
            Quantization::INT8 => 2,     // INT8
            Quantization::INT4 => 1,     // INT4
            Quantization::Custom(_) => 2, // Assume INT8 for custom
        };
        
        // Estimate number of attention heads and hidden size
        let estimated_params = base_size / 4; // Assuming FP32 parameters
        let estimated_hidden_size = (estimated_params / 1_000_000_000 * 128) as u64; // Rough estimate
        let estimated_heads = (estimated_hidden_size / 64).max(16); // Rough estimate
        
        let kv_cache_memory = (config.context_size as u64) * kv_cache_per_token * estimated_heads * 2; // 2 for K and V
        
        // Add overhead (20%)
        let total_memory = weights_memory + kv_cache_memory;
        let memory_with_overhead = total_memory + (total_memory / 5);
        
        // Adjust for parallelism
        let parallelism_factor = if config.tensor_parallel {
            1.0 / config.tensor_parallel_shards as f64
        } else {
            1.0
        };
        
        let final_memory = (memory_with_overhead as f64 * parallelism_factor) as u64;
        
        Ok(final_memory)
    }
    
    /// Validates a model configuration
    pub fn validate_config(&self, config: &ModelConfig) -> Result<()> {
        // Check if device is valid
        match &config.device {
            Device::CPU => {
                // CPU is always valid
            },
            Device::CUDA(gpu_id) => {
                // Check if CUDA is available
                if !self.hardware_capabilities.has_cuda() {
                    return Err(Error::InvalidArgument("CUDA is not available".to_string()).into());
                }
                
                // Check if GPU ID is valid
                if *gpu_id >= self.hardware_capabilities.get_gpu_count() as usize {
                    return Err(Error::InvalidArgument(format!(
                        "Invalid GPU ID: {}, only {} GPUs available",
                        gpu_id, self.hardware_capabilities.get_gpu_count()
                    )).into());
                }
            },
            Device::ROCm(gpu_id) => {
                // Check if ROCm is available
                if !self.hardware_capabilities.has_rocm() {
                    return Err(Error::InvalidArgument("ROCm is not available".to_string()).into());
                }
                
                // Check if GPU ID is valid
                if *gpu_id >= self.hardware_capabilities.get_gpu_count() as usize {
                    return Err(Error::InvalidArgument(format!(
                        "Invalid GPU ID: {}, only {} GPUs available",
                        gpu_id, self.hardware_capabilities.get_gpu_count()
                    )).into());
                }
            },
            Device::MultiGPU(gpu_ids) => {
                // Check if we have any GPUs
                if self.hardware_capabilities.get_gpu_count() == 0 {
                    return Err(Error::InvalidArgument("No GPUs available".to_string()).into());
                }
                
                // Check if all GPU IDs are valid
                for gpu_id in gpu_ids {
                    if *gpu_id >= self.hardware_capabilities.get_gpu_count() as usize {
                        return Err(Error::InvalidArgument(format!(
                            "Invalid GPU ID: {}, only {} GPUs available",
                            gpu_id, self.hardware_capabilities.get_gpu_count()
                        )).into());
                    }
                }
            },
            Device::Custom(_) => {
                // Custom device, can't validate
            },
        }
        
        // Check tensor parallelism
        if config.tensor_parallel && config.tensor_parallel_shards <= 1 {
            return Err(Error::InvalidArgument(
                "Tensor parallelism enabled but shards <= 1".to_string()
            ).into());
        }
        
        // Check pipeline parallelism
        if config.pipeline_parallel && config.pipeline_parallel_shards <= 1 {
            return Err(Error::InvalidArgument(
                "Pipeline parallelism enabled but shards <= 1".to_string()
            ).into());
        }
        
        // Check flash attention
        if config.flash_attention {
            match &config.device {
                Device::CPU => {
                    return Err(Error::InvalidArgument(
                        "Flash attention not supported on CPU".to_string()
                    ).into());
                },
                _ => {
                    // Flash attention requires GPU, which we've already validated
                }
            }
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    // These tests would require mocking the dependencies
    // For now, we just have placeholder tests
    
    #[test]
    fn test_model_config_manager() {
        // Placeholder test
        assert!(true);
    }
}

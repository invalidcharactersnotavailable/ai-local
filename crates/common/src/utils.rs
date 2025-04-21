//! Utility functions for AI Orchestrator
//!
//! This module provides utility functions used throughout the AI Orchestrator system.

use std::time::{Duration, Instant};
use std::future::Future;
use tokio::time::timeout;
use crate::error::{Error, Result};

/// Formats a byte size into a human-readable string
///
/// # Examples
///
/// ```
/// use common::utils::format_bytes;
///
/// assert_eq!(format_bytes(1024), "1.0 KiB");
/// assert_eq!(format_bytes(1048576), "1.0 MiB");
/// ```
pub fn format_bytes(bytes: u64) -> String {
    const UNITS: [&str; 6] = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"];
    
    if bytes == 0 {
        return "0 B".to_string();
    }
    
    let bytes_f64 = bytes as f64;
    let base = 1024_f64;
    let exponent = (bytes_f64.ln() / base.ln()).floor() as usize;
    let exponent = exponent.min(UNITS.len() - 1);
    
    let value = bytes_f64 / base.powi(exponent as i32);
    format!("{:.1} {}", value, UNITS[exponent])
}

/// Formats a duration into a human-readable string
///
/// # Examples
///
/// ```
/// use common::utils::format_duration;
/// use std::time::Duration;
///
/// assert_eq!(format_duration(Duration::from_secs(65)), "1m 5s");
/// ```
pub fn format_duration(duration: Duration) -> String {
    let total_secs = duration.as_secs();
    
    if total_secs == 0 {
        let millis = duration.subsec_millis();
        if millis == 0 {
            let micros = duration.subsec_micros();
            return format!("{}µs", micros);
        }
        return format!("{}ms", millis);
    }
    
    let days = total_secs / (24 * 60 * 60);
    let hours = (total_secs % (24 * 60 * 60)) / (60 * 60);
    let minutes = (total_secs % (60 * 60)) / 60;
    let seconds = total_secs % 60;
    
    let mut result = String::new();
    
    if days > 0 {
        result.push_str(&format!("{}d ", days));
    }
    
    if hours > 0 || !result.is_empty() {
        result.push_str(&format!("{}h ", hours));
    }
    
    if minutes > 0 || !result.is_empty() {
        result.push_str(&format!("{}m ", minutes));
    }
    
    result.push_str(&format!("{}s", seconds));
    
    result
}

/// Executes a future with a timeout
///
/// # Examples
///
/// ```
/// use common::utils::execute_with_timeout;
/// use std::time::Duration;
///
/// async fn example() -> Result<()> {
///     let result = execute_with_timeout(
///         async { Ok(42) },
///         Duration::from_secs(1),
///         "example operation"
///     ).await?;
///     assert_eq!(result, 42);
///     Ok(())
/// }
/// ```
pub async fn execute_with_timeout<T, F>(
    future: F,
    duration: Duration,
    operation_name: &str,
) -> Result<T>
where
    F: Future<Output = Result<T>>,
{
    match timeout(duration, future).await {
        Ok(result) => result,
        Err(_) => Err(Error::Timeout(format!("Operation '{}' timed out after {}", 
                                            operation_name, 
                                            format_duration(duration)))),
    }
}

/// Measures the execution time of a function
///
/// # Examples
///
/// ```
/// use common::utils::measure_execution_time;
///
/// fn example() -> Result<()> {
///     let (result, duration) = measure_execution_time(|| {
///         // Some operation
///         Ok(42)
///     })?;
///     assert_eq!(result, 42);
///     println!("Execution time: {:?}", duration);
///     Ok(())
/// }
/// ```
pub fn measure_execution_time<T, F>(f: F) -> Result<(T, Duration)>
where
    F: FnOnce() -> Result<T>,
{
    let start = Instant::now();
    let result = f()?;
    let duration = start.elapsed();
    Ok((result, duration))
}

/// Measures the execution time of an async function
///
/// # Examples
///
/// ```
/// use common::utils::measure_execution_time_async;
///
/// async fn example() -> Result<()> {
///     let (result, duration) = measure_execution_time_async(async {
///         // Some async operation
///         Ok(42)
///     }).await?;
///     assert_eq!(result, 42);
///     println!("Execution time: {:?}", duration);
///     Ok(())
/// }
/// ```
pub async fn measure_execution_time_async<T, F>(future: F) -> Result<(T, Duration)>
where
    F: Future<Output = Result<T>>,
{
    let start = Instant::now();
    let result = future.await?;
    let duration = start.elapsed();
    Ok((result, duration))
}

/// Truncates a string to a maximum length, adding an ellipsis if truncated
///
/// # Examples
///
/// ```
/// use common::utils::truncate_string;
///
/// assert_eq!(truncate_string("Hello, world!", 5), "Hello...");
/// assert_eq!(truncate_string("Hello", 10), "Hello");
/// ```
pub fn truncate_string(s: &str, max_length: usize) -> String {
    if s.len() <= max_length {
        s.to_string()
    } else {
        format!("{}...", &s[..max_length])
    }
}

/// Parses a string into a boolean value
///
/// # Examples
///
/// ```
/// use common::utils::parse_bool;
///
/// assert_eq!(parse_bool("true"), true);
/// assert_eq!(parse_bool("yes"), true);
/// assert_eq!(parse_bool("1"), true);
/// assert_eq!(parse_bool("false"), false);
/// assert_eq!(parse_bool("no"), false);
/// assert_eq!(parse_bool("0"), false);
/// ```
pub fn parse_bool(s: &str) -> bool {
    match s.to_lowercase().as_str() {
        "true" | "yes" | "y" | "1" | "on" | "enable" | "enabled" => true,
        _ => false,
    }
}

/// Returns the number of CPU cores available
pub fn get_num_cpus() -> usize {
    num_cpus::get()
}

/// Returns the number of physical CPU cores available
pub fn get_num_physical_cpus() -> usize {
    num_cpus::get_physical()
}

/// Checks if CUDA is available
pub fn is_cuda_available() -> bool {
    #[cfg(feature = "cuda")]
    {
        // Simple check for CUDA availability
        unsafe {
            let mut device_count = 0;
            let result = cuda_runtime_sys::cudaGetDeviceCount(&mut device_count as *mut _);
            result == 0 && device_count > 0
        }
    }
    
    #[cfg(not(feature = "cuda"))]
    {
        false
    }
}

/// Checks if ROCm is available
pub fn is_rocm_available() -> bool {
    #[cfg(feature = "rocm")]
    {
        // Simple check for ROCm availability
        // This is a placeholder and would need actual implementation
        false
    }
    
    #[cfg(not(feature = "rocm"))]
    {
        false
    }
}

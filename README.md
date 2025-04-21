# AI Orchestrator in Rust

A high-performance AI orchestrator system built in Rust, designed to handle AI models from 1B to 500B+ parameters across diverse hardware configurations.

## Features

- **Modular Architecture**: Handles AI models from 1B to 500B+ parameters
- **Performance Optimized**: Built in Rust with advanced performance optimizations
- **Resource Management**: Efficiently manages CPU, GPU, and memory resources
- **Task Scheduling**: Priority-based task scheduling with dependency resolution
- **Search Functionality**: Powerful search capabilities with both keyword and vector search
- **Advanced Reasoning**: Scaling inference-time compute based on task complexity
- **Hardware Adaptability**: Works across diverse hardware from 2GB RAM systems to data centers
- **Web Interface**: Google-style UI with model recommendations and task management
- **Environment Check**: Automatic hardware profiling on startup
- **Model Recommendations**: Suggests models that will run smoothly on your hardware

## Project Structure

```
ai_orchestrator/
├── crates/                      # Modular components
│   ├── advanced-reasoning/      # Advanced reasoning capabilities
│   ├── api-gateway/             # API gateway for external access
│   ├── cli-interface/           # Command-line interface
│   ├── common/                  # Shared types and utilities
│   ├── config/                  # Configuration management
│   ├── hardware-profiler/       # Hardware detection and profiling
│   ├── inference-engine/        # Model inference execution
│   ├── logging/                 # Logging infrastructure
│   ├── model-manager/           # Model management and loading
│   ├── orchestrator-core/       # Core orchestration engine
│   ├── performance-monitor/     # Performance monitoring
│   ├── performance-optimizations/ # Performance optimization utilities
│   ├── resource-manager/        # System resource management
│   ├── scaling-adapter/         # Scaling for different hardware
│   ├── search-engine/           # Search functionality
│   ├── security/                # Security features
│   ├── storage-adapter/         # Storage management
│   ├── task-scheduler/          # Task scheduling and execution
│   └── web-interface/           # Web interface
└── src/                         # Main application
    ├── lib.rs                   # Library interface
    └── main.rs                  # Application entry point
```

## Performance Optimizations

- **Memory Pools**: Custom memory allocation for reduced overhead
- **SIMD Acceleration**: Vector and matrix operations using CPU SIMD instructions
- **Lock-Free Data Structures**: Concurrent access without locks
- **Batch Processing**: Efficient batch processing of tasks
- **Async I/O**: Non-blocking I/O operations
- **Zero-Copy**: Minimized data copying
- **Custom Memory Management**: Specialized for AI workloads

## Advanced Reasoning

The system supports different reasoning complexity levels:

- **Simple**: Fast, low compute usage
- **Standard**: Balanced performance and quality
- **Advanced**: Higher quality, more compute intensive
- **Expert**: Maximum quality, highest compute usage

The system can automatically determine the appropriate complexity level based on task requirements, or you can manually specify it.

## Web Interface

The application includes a web interface that:

- Automatically opens in your browser on startup
- Provides a clean, Google-style UI
- Shows system status and available resources
- Presents a curated list of recommended AI models based on your hardware
- Allows you to select which models to download
- Lets you choose between different reasoning modes
- Provides task management and monitoring

## Environment Check

On every startup, the application:

- Profiles your hardware (CPU, RAM, GPU)
- Determines optimal configurations
- Adapts to any hardware changes
- Provides compatibility ratings for available models
- Recommends models that will run smoothly on your system

## Building and Running

### Prerequisites

- Rust 1.70+ (2021 edition)
- Cargo
- Optional: CUDA for GPU support

### Building

```bash
# Clone the repository
git clone https://github.com/yourusername/ai_orchestrator.git
cd ai_orchestrator

# Build in release mode
cargo build --release
```

### Running

```bash
# Run the main application
cargo run --release
```

When you run the application:
1. It will perform an environment check
2. Start the AI orchestrator
3. Open a web browser to the interface
4. Show model recommendations based on your hardware

## Usage Examples

### Using the Web Interface

1. **Start the application**:
   ```bash
   cargo run --release
   ```

2. **Browser opens automatically** to http://localhost:80

3. **View model recommendations**:
   - Models are sorted by compatibility with your hardware
   - "Recommended" tags highlight models that will run smoothly
   - Select models to download

4. **Choose reasoning mode**:
   - Simple: Fast responses with minimal compute
   - Standard: Balanced performance and quality
   - Advanced: Higher quality reasoning with more compute
   - Expert: Maximum quality with full compute scaling

5. **Submit and monitor tasks**:
   - Create new tasks through the web interface
   - Monitor progress of running tasks
   - View results of completed tasks

### Programmatic API Usage

```rust
use ai_orchestrator::AIOrchestrator;
use common::models::{TaskType, TaskPriority};
use task_scheduler::TaskDefinition;
use advanced_reasoning::{AdvancedReasoningEngine, ReasoningTask, ReasoningComplexity};

#[tokio::main]
async fn main() -> Result<()> {
    // Create and start the orchestrator
    let orchestrator = AIOrchestrator::new().await?;
    orchestrator.start().await?;
    
    // Get the reasoning engine
    let reasoning_engine = Arc::new(AdvancedReasoningEngine::new(
        orchestrator.get_config_manager(),
        orchestrator.get_model_manager(),
        orchestrator.get_resource_manager(),
        orchestrator.get_performance_optimizer(),
    )?);
    
    // Create a reasoning task
    let task = ReasoningTask {
        id: "complex-task".to_string(),
        input: "Explain quantum mechanics".to_string(),
        complexity: ReasoningComplexity::Expert,
        model: "sample-model".to_string(),
        max_tokens: 1000,
        temperature: 0.7,
        parameters: HashMap::new(),
    };
    
    // Submit task
    reasoning_engine.submit_task(task.clone()).await?;
    
    // Get results
    let result = reasoning_engine.get_result(&task.id).await?;
    println!("Output: {}", result.output);
}
```

## Configuration

The system can be configured through environment variables or a configuration file. Key configuration options:

- `MAX_CONCURRENT_TASKS`: Maximum number of concurrent tasks
- `DEFAULT_REASONING_COMPLEXITY`: Default complexity level for reasoning
- `AUTO_SCALING_ENABLED`: Whether to automatically scale compute based on task complexity
- `MEMORY_POOL_SIZE`: Size of memory pools for optimized allocation
- `WEB_INTERFACE_PORT`: Port for the web interface (default: 80)

## License

MIT License

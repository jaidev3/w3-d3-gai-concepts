# LLM Inference Calculator

A comprehensive tool for estimating Large Language Model (LLM) inference costs, latency, and memory usage across different hardware configurations and deployment scenarios.

## 🎯 Overview

This calculator helps you make informed decisions about LLM deployment by providing accurate estimates for:
- **Memory Usage**: VRAM requirements including model weights, KV cache, and overhead
- **Latency**: End-to-end inference time including prefill and decode phases  
- **Cost**: Per-request and per-token pricing based on hardware and deployment mode
- **Hardware Compatibility**: Whether your hardware can handle the model
- **Optimization Recommendations**: Actionable suggestions to improve efficiency

## 🚀 Features

### Supported Models
- **7B Parameters**: Efficient models like Mistral 7B, Llama 2 7B
- **13B Parameters**: Balanced performance models like Llama 2 13B
- **GPT-4 Scale**: Large-scale models with 1.8T parameters

### Hardware Configurations
- RTX 4090 (24GB) - Consumer/prosumer
- A10G (24GB) - Cloud optimized
- A100 40GB/80GB - Enterprise
- H100 (80GB) - Latest generation
- CPU-only - Fallback option

### Deployment Modes
- Single GPU
- Multi-GPU
- Distributed (4+ GPUs)
- Cloud API

### Precision Options
- FP16 (full precision)
- INT8 (quantized)
- INT4 (highly quantized)

## 📥 Installation

```bash
# Clone the repository
git clone <repository-url>
cd q1

# No additional dependencies required - uses Python standard library only
python inference_calculator.py
```

## 🔧 Usage

### Basic Usage

```python
from inference_calculator import LLMInferenceCalculator, ModelSize, HardwareType, DeploymentMode

# Initialize calculator
calculator = LLMInferenceCalculator()

# Calculate inference metrics
result = calculator.calculate_inference(
    model_size=ModelSize.SMALL_7B,
    input_tokens=100,
    output_tokens=200,
    batch_size=1,
    hardware_type=HardwareType.A100_40GB,
    deployment_mode=DeploymentMode.SINGLE_GPU,
    precision="fp16"
)

# View results
print(f"Memory Usage: {result.memory_usage_gb:.2f} GB")
print(f"Latency: {result.latency_ms:.1f} ms")
print(f"Cost per Request: ${result.cost_per_request:.6f}")
print(f"Compatible: {result.hardware_compatible}")
```

### Advanced Usage

```python
# Compare different configurations
configs = [
    {"model": ModelSize.SMALL_7B, "hardware": HardwareType.RTX_4090},
    {"model": ModelSize.MEDIUM_13B, "hardware": HardwareType.A100_80GB},
    {"model": ModelSize.LARGE_GPT4, "hardware": HardwareType.H100}
]

for config in configs:
    result = calculator.calculate_inference(
        model_size=config["model"],
        input_tokens=500,
        output_tokens=1000,
        hardware_type=config["hardware"]
    )
    print(f"{config['model'].value}: ${result.cost_per_request:.6f}")
```

## 📊 Example Output

```
🧮 LLM Inference Calculator
==================================================

1. 7B Model - Single A100
------------------------------
💾 Memory Usage: 16.88 GB
⏱️  Latency: 2857.6 ms
💰 Cost per Request: $0.000397
💸 Cost per Token: $0.00000397
🚀 Tokens/Second: 35.0
🔧 Compatible: ✅
📊 Memory Utilization: 42.2%

📋 Recommendations:
  ✅ Efficient choice for most applications
  💡 Consider int8 quantization for better efficiency
```

## 🧮 Calculator Components

### Core Classes

#### `LLMInferenceCalculator`
Main calculator class with methods for:
- `calculate_memory_usage()`: Estimates VRAM requirements
- `calculate_latency()`: Computes prefill + decode latency
- `calculate_cost()`: Determines per-request pricing
- `check_hardware_compatibility()`: Validates hardware requirements
- `generate_recommendations()`: Provides optimization suggestions

#### `InferenceResult`
Dataclass containing all calculation results:
```python
@dataclass
class InferenceResult:
    latency_ms: float
    memory_usage_gb: float
    cost_per_request: float
    cost_per_token: float
    tokens_per_second: float
    hardware_compatible: bool
    memory_utilization: float
    recommendations: List[str]
```

### Memory Calculation

The calculator estimates memory usage including:

1. **Base Model Memory**: Model weights in specified precision
2. **KV Cache**: Key-Value cache scaling with sequence length
3. **Overhead**: Activations, gradients, and system overhead (~20%)

```python
total_memory = base_memory + kv_cache + overhead
```

### Latency Calculation

Accounts for both phases of transformer inference:

1. **Prefill Phase**: Parallel processing of input tokens
2. **Decode Phase**: Sequential generation of output tokens

```python
total_latency = prefill_latency + decode_latency
```

### Cost Calculation

Based on hardware hourly rates and actual inference time:

```python
cost_per_request = hardware_hourly_rate * (latency_ms / 3600000)
```

## 📈 Scenario Analysis

The project includes analysis of three real-world scenarios:

1. **Customer Support Chatbot** (7B model)
   - 1,000 requests/day
   - Real-time responses required
   - Budget: <$500/month

2. **Content Generation Platform** (13B model)
   - 200 requests/day
   - High-quality content
   - Budget: <$2,000/month

3. **Research Institution** (GPT-4 scale)
   - 50 complex analyses/day
   - Maximum quality required
   - Budget: $5,000/month

See [`scenario_analysis.md`](scenario_analysis.md) for detailed analysis and recommendations.

## 🔬 Research Foundation

The calculator is built on comprehensive research of LLM inference:

- **Model Architectures**: Transformer mechanics, attention mechanisms, KV caching
- **Performance Benchmarks**: Actual measurements from 7B, 13B, and GPT-4 scale models
- **Hardware Specifications**: Real-world GPU memory, compute, and pricing data
- **Optimization Techniques**: Quantization, batching, and deployment strategies

See [`research_notes.md`](research_notes.md) for the complete research foundation.

## ⚡ Optimization Recommendations

The calculator provides intelligent recommendations:

### Memory Optimization
- **Quantization**: INT8/INT4 to reduce memory by 50-75%
- **Gradient Checkpointing**: Reduce activation memory
- **Model Sharding**: Distribute across multiple GPUs

### Performance Optimization  
- **Batch Processing**: Improve throughput for non-real-time applications
- **KV Cache Optimization**: Manage memory vs. compute trade-offs
- **Hardware Upgrades**: Cost-effective performance improvements

### Cost Optimization
- **Model Right-sizing**: Match model capability to use case requirements
- **Cloud vs. On-premise**: Consider operational vs. infrastructure costs
- **Hybrid Deployments**: Use different models for different quality needs

## 🎯 Use Cases

### Ideal For
- **Deployment Planning**: Choose optimal hardware configuration
- **Cost Estimation**: Budget for LLM infrastructure
- **Performance Tuning**: Identify bottlenecks and optimization opportunities
- **Hardware Procurement**: Make informed GPU purchasing decisions
- **Architecture Design**: Plan scalable LLM systems

### Not Suitable For
- Production monitoring (estimates only)
- Fine-tuning cost calculation
- Training infrastructure planning
- Real-time dynamic optimization

## 🤝 Contributing

Contributions welcome! Areas for improvement:

1. **Additional Models**: Support for more model architectures
2. **Hardware Expansion**: Add support for new GPU generations
3. **Optimization Techniques**: More sophisticated memory/latency models
4. **Cloud Providers**: Integrate with AWS, GCP, Azure pricing APIs
5. **UI/Web Interface**: Build interactive calculator interface

## 📄 License

This project is provided for educational and research purposes. See LICENSE file for details.

## 🔗 Related Resources

- [Transformer Inference Optimization](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/)
- [LLM Memory Requirements](https://modal.com/blog/how-much-vram-need-inference)
- [KV Cache Techniques](https://huggingface.co/blog/not-lain/kv-caching)
- [Model Performance Benchmarks](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)

---

*Built with comprehensive research and real-world testing data. See research_notes.md for detailed technical foundation.*

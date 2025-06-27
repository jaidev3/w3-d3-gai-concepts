# LLM Inference Calculator - User Interfaces

This directory contains multiple interfaces for the LLM Inference Calculator, allowing you to calculate memory usage, latency, and costs for LLM inference deployments.

## 🖥️ Available Interfaces

### 1. Command Line Interface (CLI) - `calculator_cli.py`
**Recommended for: Interactive exploration and learning**

An interactive terminal-based interface with guided menus and validation.

**Features:**
- 🎯 Step-by-step guided configuration
- 🔍 Hardware comparison mode
- 📖 Built-in help and explanations
- ✅ Input validation and error handling
- 🎨 Clean, formatted output

**Usage:**
```bash
python calculator_cli.py
```

**Screenshots:**
```
🧮 LLM Inference Calculator
Calculate memory, latency, and costs for LLM inference
============================================================

📋 Main Menu
-----------
1. 🧮 Calculate Inference Metrics
2. 🔍 Hardware Comparison
3. 📖 Quick Guide
4. ℹ️ About
0. 🚪 Exit

Enter your choice (0-4):
```

### 2. Web Interface (Streamlit) - `calculator_ui.py`
**Recommended for: Visual exploration and presentations**

Modern web-based interface with interactive widgets and visualizations.

**Features:**
- 🎨 Modern, responsive web interface
- 📊 Interactive charts and visualizations
- 🔄 Real-time calculations
- 📈 Cost projection graphs
- 🔍 Hardware comparison tables
- 💾 Exportable results

**Setup:**
```bash
# Install dependencies
pip install -r requirements.txt

# Run the web interface
streamlit run calculator_ui.py
```

**Note:** Requires additional dependencies (Streamlit, Plotly, Pandas, NumPy)

### 3. Core Calculator - `inference_calculator.py`
**Recommended for: Advanced customization and integration**

The underlying calculation engine with full control over all parameters.

**Features:**
- 🔧 Complete control over all parameters
- 📊 Detailed calculation methods
- 🎯 Enum-based type safety
- 💡 Built-in recommendations engine
- 🧮 Advanced memory and latency modeling

**Usage:**
```python
from inference_calculator import LLMInferenceCalculator, ModelSize, HardwareType

calculator = LLMInferenceCalculator()
result = calculator.calculate_inference(
    model_size=ModelSize.SMALL_7B,
    input_tokens=150,
    output_tokens=100,
    hardware_type=HardwareType.RTX_4090
)
```

## 🚀 Quick Start Examples

### Example 1: Basic Calculation with Core Calculator
```python
from inference_calculator import LLMInferenceCalculator, ModelSize, HardwareType

calculator = LLMInferenceCalculator()
result = calculator.calculate_inference(
    model_size=ModelSize.SMALL_7B,
    input_tokens=150,
    output_tokens=100,
    hardware_type=HardwareType.RTX_4090
)

print(f"Memory usage: {result.memory_usage_gb:.2f} GB")
print(f"Latency: {result.latency_ms:.1f} ms")
print(f"Cost per request: ${result.cost_per_request:.6f}")
```

### Example 2: Using the Web Interface
```bash
# Install dependencies
pip install -r requirements.txt

# Launch the web interface
streamlit run calculator_ui.py
```

### Example 3: Using the CLI Interface
```bash
# Launch interactive CLI
python calculator_cli.py
```

## 📊 Understanding the Results

### Key Metrics Explained

**Memory Usage:**
- Model weights + KV cache + processing overhead
- Varies by model size, precision, and sequence length
- Critical for hardware compatibility

**Latency:**
- Prefill time (processing input) + decode time (generating output)
- Measured in milliseconds
- Lower is better for real-time applications

**Cost:**
- Based on hardware pricing and inference time
- Includes deployment mode multipliers
- Useful for budget planning

**Throughput:**
- Tokens generated per second
- Higher is better for batch processing
- Affected by hardware and model size

### Precision Trade-offs

| Precision | Memory | Speed | Quality | Use Case |
|-----------|--------|-------|---------|----------|
| FP16 | 1x | Baseline | Best | Research, premium applications |
| INT8 | 0.5x | ~1.5x faster | Very Good | Production deployments |
| INT4 | 0.25x | ~2x faster | Good | High-throughput applications |

### Hardware Recommendations

| Use Case | Model | Hardware | Reasoning |
|----------|-------|----------|-----------|
| Chatbot | 7B + INT8 | RTX 4090 | Cost-effective, good quality |
| Content Gen | 13B + FP16 | A100 80GB | Quality-focused, batch capable |
| Research | GPT-4 | Cloud API | Highest quality, pay-per-use |
| Code Assistant | 13B + FP16 | A100 40GB | Good reasoning, reasonable latency |

## 🔧 Advanced Configuration

### Deployment Modes
- **Single GPU**: Most common, cost-effective
- **Multi-GPU**: 2x cost, better for larger models
- **Distributed**: 4x cost, required for largest models
- **Cloud API**: Variable cost, no infrastructure needed

### Batch Processing
- Batch size > 1 can improve throughput
- Increases latency but reduces cost per token
- Best for non-real-time applications

### Cost Optimization Tips
1. **Start with 7B models** - good quality, low cost
2. **Use INT8 quantization** - 2x memory reduction, minimal quality loss
3. **Batch similar requests** - improves efficiency
4. **Consider cloud APIs** - for occasional high-quality needs
5. **Monitor utilization** - ensure hardware is well-utilized

## 🐛 Troubleshooting

### Common Issues

**"Hardware incompatible" error:**
- Model requires more memory than hardware has
- Try smaller model, INT8/INT4 precision, or better hardware

**Very high latency:**
- CPU-only inference is slow - use GPU
- Large models need powerful hardware
- Consider smaller models for real-time use

**High costs:**
- Reduce model size or use quantization
- Increase batch size for batch workloads
- Consider cloud APIs for low-volume use

### Getting Help

1. **Use the CLI Quick Guide**: `python calculator_cli.py` → option 3
2. **Check recommendations**: All interfaces provide optimization suggestions
3. **Review scenario analysis**: See `scenario_analysis.md` for real-world examples
4. **Experiment with parameters**: Try different configurations to find optimal settings

## 📈 Next Steps

1. **Try the interfaces**: Start with CLI for learning, use simple functions for integration
2. **Analyze your use case**: Use scenario analysis to find similar applications
3. **Optimize configuration**: Follow recommendations and test different options
4. **Plan deployment**: Use cost projections for budgeting and capacity planning
5. **Monitor performance**: Track actual vs predicted metrics in production 
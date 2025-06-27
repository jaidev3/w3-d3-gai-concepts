#!/usr/bin/env python3
"""
LLM Inference Calculator
Estimates costs, latency, and memory usage for LLM inference.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from enum import Enum


class ModelSize(Enum):
    """Supported model sizes"""
    SMALL_7B = "7B"
    MEDIUM_13B = "13B"
    LARGE_GPT4 = "GPT-4"


class HardwareType(Enum):
    """Supported hardware configurations"""
    RTX_4090 = "RTX 4090"
    A10G = "A10G"
    A100_40GB = "A100 40GB"
    A100_80GB = "A100 80GB"
    H100 = "H100"
    CPU_ONLY = "CPU Only"


class DeploymentMode(Enum):
    """Deployment modes"""
    SINGLE_GPU = "single_gpu"
    MULTI_GPU = "multi_gpu"
    DISTRIBUTED = "distributed"
    CLOUD_API = "cloud_api"


@dataclass
class HardwareSpec:
    """Hardware specifications"""
    memory_gb: float
    compute_capability: float
    price_per_hour: float
    tokens_per_second: float
    power_watts: int


@dataclass
class ModelSpec:
    """Model specifications"""
    parameters: int
    memory_fp16_gb: float
    memory_int8_gb: float
    memory_int4_gb: float
    base_latency_ms: float
    compute_intensity: float


@dataclass
class InferenceResult:
    """Results from inference calculation"""
    latency_ms: float
    memory_usage_gb: float
    cost_per_request: float
    cost_per_token: float
    tokens_per_second: float
    hardware_compatible: bool
    memory_utilization: float
    recommendations: List[str]


class LLMInferenceCalculator:
    """Main calculator class for LLM inference metrics"""
    
    def __init__(self):
        self.hardware_specs = self._init_hardware_specs()
        self.model_specs = self._init_model_specs()
        
    def _init_hardware_specs(self) -> Dict[HardwareType, HardwareSpec]:
        """Initialize hardware specifications based on research"""
        return {
            HardwareType.RTX_4090: HardwareSpec(
                memory_gb=24,
                compute_capability=8.9,
                price_per_hour=0.50,  # Estimated cloud cost
                tokens_per_second=35,
                power_watts=450
            ),
            HardwareType.A10G: HardwareSpec(
                memory_gb=24,
                compute_capability=8.6,
                price_per_hour=0.75,
                tokens_per_second=30,
                power_watts=300
            ),
            HardwareType.A100_40GB: HardwareSpec(
                memory_gb=40,
                compute_capability=8.0,
                price_per_hour=3.20,
                tokens_per_second=50,
                power_watts=400
            ),
            HardwareType.A100_80GB: HardwareSpec(
                memory_gb=80,
                compute_capability=8.0,
                price_per_hour=4.10,
                tokens_per_second=55,
                power_watts=400
            ),
            HardwareType.H100: HardwareSpec(
                memory_gb=80,
                compute_capability=9.0,
                price_per_hour=8.00,
                tokens_per_second=80,
                power_watts=700
            ),
            HardwareType.CPU_ONLY: HardwareSpec(
                memory_gb=128,  # Typical server RAM
                compute_capability=0.0,
                price_per_hour=0.20,
                tokens_per_second=2,
                power_watts=200
            )
        }
    
    def _init_model_specs(self) -> Dict[ModelSize, ModelSpec]:
        """Initialize model specifications based on research"""
        return {
            ModelSize.SMALL_7B: ModelSpec(
                parameters=7_000_000_000,
                memory_fp16_gb=14,
                memory_int8_gb=7,
                memory_int4_gb=3.5,
                base_latency_ms=50,
                compute_intensity=1.0
            ),
            ModelSize.MEDIUM_13B: ModelSpec(
                parameters=13_000_000_000,
                memory_fp16_gb=26,
                memory_int8_gb=13,
                memory_int4_gb=6.5,
                base_latency_ms=75,
                compute_intensity=1.8
            ),
            ModelSize.LARGE_GPT4: ModelSpec(
                parameters=1_800_000_000_000,  # 1.8T parameters
                memory_fp16_gb=500,  # Distributed across multiple GPUs
                memory_int8_gb=250,
                memory_int4_gb=125,
                base_latency_ms=200,
                compute_intensity=10.0
            )
        }
    
    def calculate_memory_usage(self, 
                             model_size: ModelSize, 
                             batch_size: int,
                             max_tokens: int,
                             precision: str = "fp16") -> float:
        """Calculate memory usage for inference"""
        model_spec = self.model_specs[model_size]
        
        # Base model memory
        if precision == "fp16":
            base_memory = model_spec.memory_fp16_gb
        elif precision == "int8":
            base_memory = model_spec.memory_int8_gb
        elif precision == "int4":
            base_memory = model_spec.memory_int4_gb
        else:
            base_memory = model_spec.memory_fp16_gb
        
        # KV cache memory (grows with sequence length and batch size)
        # More accurate approximation: KV cache scales with hidden dimensions, not total parameters
        # Typical KV cache is ~0.1-0.5% of model size per token for transformers
        kv_cache_factor = 0.002  # 0.2% of model memory per token
        kv_cache_gb = base_memory * kv_cache_factor * max_tokens * batch_size
        
        # Additional overhead (activations, gradients, etc.)
        overhead_gb = base_memory * 0.2
        
        total_memory = base_memory + kv_cache_gb + overhead_gb
        return total_memory
    
    def calculate_latency(self,
                         model_size: ModelSize,
                         input_tokens: int,
                         output_tokens: int,
                         batch_size: int,
                         hardware: HardwareType) -> Tuple[float, float]:
        """Calculate inference latency (prefill + decode phases)"""
        model_spec = self.model_specs[model_size]
        hardware_spec = self.hardware_specs[hardware]
        
        # Prefill latency (parallel processing of input)
        prefill_latency = (model_spec.base_latency_ms * 
                          (input_tokens / 100) * 
                          model_spec.compute_intensity /
                          max(hardware_spec.compute_capability, 0.1))
        
        # Decode latency (sequential token generation)
        decode_latency = (output_tokens / hardware_spec.tokens_per_second) * 1000
        
        # Batch processing can improve throughput but increases latency
        batch_factor = 1 + (batch_size - 1) * 0.1
        
        total_latency = (prefill_latency + decode_latency) * batch_factor
        return total_latency, prefill_latency
    
    def calculate_cost(self,
                      hardware: HardwareType,
                      latency_ms: float,
                      deployment_mode: DeploymentMode) -> float:
        """Calculate cost per request"""
        hardware_spec = self.hardware_specs[hardware]
        
        # Convert latency to hours
        latency_hours = latency_ms / (1000 * 60 * 60)
        
        # Base cost calculation
        base_cost = hardware_spec.price_per_hour * latency_hours
        
        # Deployment mode adjustments
        if deployment_mode == DeploymentMode.MULTI_GPU:
            base_cost *= 2  # Assume 2 GPUs
        elif deployment_mode == DeploymentMode.DISTRIBUTED:
            base_cost *= 4  # Assume 4 GPUs
        elif deployment_mode == DeploymentMode.CLOUD_API:
            base_cost *= 1.5  # Cloud provider markup
        
        return base_cost
    
    def check_hardware_compatibility(self,
                                   required_memory: float,
                                   hardware: HardwareType) -> Tuple[bool, float]:
        """Check if hardware can handle the model"""
        hardware_spec = self.hardware_specs[hardware]
        compatible = required_memory <= hardware_spec.memory_gb
        utilization = required_memory / hardware_spec.memory_gb
        return compatible, utilization
    
    def generate_recommendations(self,
                               model_size: ModelSize,
                               hardware: HardwareType,
                               memory_usage: float,
                               compatible: bool,
                               utilization: float) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []
        
        if not compatible:
            recommendations.append(
                f"⚠️ Hardware incompatible: {memory_usage:.1f}GB required, "
                f"{self.hardware_specs[hardware].memory_gb}GB available"
            )
            recommendations.append("Consider: Upgrade hardware or use quantization")
        
        if utilization > 0.9:
            recommendations.append("⚠️ High memory utilization (>90%) - consider optimizations")
        
        if model_size == ModelSize.SMALL_7B:
            recommendations.append("✅ Efficient choice for most applications")
            recommendations.append("💡 Consider int8 quantization for better efficiency")
        elif model_size == ModelSize.MEDIUM_13B:
            recommendations.append("⚖️ Good balance of performance and resources")
            recommendations.append("💡 Consider multi-GPU setup for better throughput")
        elif model_size == ModelSize.LARGE_GPT4:
            recommendations.append("🎯 Maximum performance but high resource requirements")
            recommendations.append("💡 Distributed deployment recommended")
        
        if hardware == HardwareType.CPU_ONLY:
            recommendations.append("🐌 CPU inference is very slow - GPU recommended")
        
        return recommendations
    
    def calculate_inference(self,
                          model_size: ModelSize,
                          input_tokens: int,
                          output_tokens: int,
                          batch_size: int = 1,
                          hardware_type: HardwareType = HardwareType.A100_40GB,
                          deployment_mode: DeploymentMode = DeploymentMode.SINGLE_GPU,
                          precision: str = "fp16") -> InferenceResult:
        """Main calculation method"""
        
        # Calculate memory usage
        max_tokens = input_tokens + output_tokens
        memory_usage = self.calculate_memory_usage(
            model_size, batch_size, max_tokens, precision
        )
        
        # Calculate latency
        total_latency, prefill_latency = self.calculate_latency(
            model_size, input_tokens, output_tokens, batch_size, hardware_type
        )
        
        # Calculate costs
        cost_per_request = self.calculate_cost(
            hardware_type, total_latency, deployment_mode
        )
        cost_per_token = cost_per_request / output_tokens if output_tokens > 0 else 0
        
        # Check hardware compatibility
        compatible, utilization = self.check_hardware_compatibility(
            memory_usage, hardware_type
        )
        
        # Calculate tokens per second
        tokens_per_second = (output_tokens / (total_latency / 1000)) if total_latency > 0 else 0
        
        # Generate recommendations
        recommendations = self.generate_recommendations(
            model_size, hardware_type, memory_usage, compatible, utilization
        )
        
        return InferenceResult(
            latency_ms=total_latency,
            memory_usage_gb=memory_usage,
            cost_per_request=cost_per_request,
            cost_per_token=cost_per_token,
            tokens_per_second=tokens_per_second,
            hardware_compatible=compatible,
            memory_utilization=utilization,
            recommendations=recommendations
        )


def main():
    """Example usage and testing"""
    calculator = LLMInferenceCalculator()
    
    print("🧮 LLM Inference Calculator")
    print("=" * 50)
    
    # Example calculations
    test_cases = [
        {
            "name": "7B Model - Single A100",
            "model_size": ModelSize.SMALL_7B,
            "input_tokens": 100,
            "output_tokens": 200,
            "batch_size": 1,
            "hardware": HardwareType.A100_40GB,
            "deployment": DeploymentMode.SINGLE_GPU,
            "precision": "fp16"
        },
        {
            "name": "13B Model - Multi-GPU",
            "model_size": ModelSize.MEDIUM_13B,
            "input_tokens": 500,
            "output_tokens": 1000,
            "batch_size": 4,
            "hardware": HardwareType.A100_80GB,
            "deployment": DeploymentMode.MULTI_GPU,
            "precision": "int8"
        },
        {
            "name": "GPT-4 - Distributed",
            "model_size": ModelSize.LARGE_GPT4,
            "input_tokens": 1000,
            "output_tokens": 500,
            "batch_size": 1,
            "hardware": HardwareType.H100,
            "deployment": DeploymentMode.DISTRIBUTED,
            "precision": "fp16"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. {test_case['name']}")
        print("-" * 30)
        
        result = calculator.calculate_inference(
            model_size=test_case["model_size"],
            input_tokens=test_case["input_tokens"],
            output_tokens=test_case["output_tokens"],
            batch_size=test_case["batch_size"],
            hardware_type=test_case["hardware"],
            deployment_mode=test_case["deployment"],
            precision=test_case["precision"]
        )
        
        print(f"💾 Memory Usage: {result.memory_usage_gb:.2f} GB")
        print(f"⏱️  Latency: {result.latency_ms:.1f} ms")
        print(f"💰 Cost per Request: ${result.cost_per_request:.6f}")
        print(f"💸 Cost per Token: ${result.cost_per_token:.8f}")
        print(f"🚀 Tokens/Second: {result.tokens_per_second:.1f}")
        print(f"🔧 Compatible: {'✅' if result.hardware_compatible else '❌'}")
        print(f"📊 Memory Utilization: {result.memory_utilization:.1%}")
        
        print("\n📋 Recommendations:")
        for rec in result.recommendations:
            print(f"  {rec}")


if __name__ == "__main__":
    main()

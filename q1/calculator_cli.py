#!/usr/bin/env python3
"""
LLM Inference Calculator - Command Line Interface
Interactive CLI for calculating LLM inference metrics
"""

import os
import sys
from typing import List, Dict, Tuple, Optional
from inference_calculator import (
    LLMInferenceCalculator, 
    ModelSize, 
    HardwareType, 
    DeploymentMode,
    InferenceResult
)

class CalculatorCLI:
    """Command line interface for the LLM Inference Calculator"""
    
    def __init__(self):
        self.calculator = LLMInferenceCalculator()
        self.clear_screen = 'cls' if os.name == 'nt' else 'clear'
        
    def clear(self):
        """Clear the terminal screen"""
        os.system(self.clear_screen)
    
    def print_header(self):
        """Print the application header"""
        print("=" * 60)
        print("🧮 LLM Inference Calculator")
        print("Calculate memory, latency, and costs for LLM inference")
        print("=" * 60)
        print()
    
    def print_menu(self, title: str, options: List[str], include_back: bool = True):
        """Print a menu with options"""
        print(f"\n📋 {title}")
        print("-" * len(title))
        
        for i, option in enumerate(options, 1):
            print(f"{i}. {option}")
        
        if include_back:
            print(f"{len(options) + 1}. ← Back")
            print("0. 🚪 Exit")
        else:
            print("0. 🚪 Exit")
        print()
    
    def get_user_choice(self, max_option: int, include_back: bool = True) -> int:
        """Get user choice with validation"""
        max_val = max_option + 1 if include_back else max_option
        
        while True:
            try:
                choice = input(f"Enter your choice (0-{max_val}): ").strip()
                choice = int(choice)
                
                if 0 <= choice <= max_val:
                    return choice
                else:
                    print(f"❌ Please enter a number between 0 and {max_val}")
            except ValueError:
                print("❌ Please enter a valid number")
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                sys.exit(0)
    
    def get_number_input(self, prompt: str, min_val: int = 1, max_val: int = None, default: int = None) -> int:
        """Get numerical input with validation"""
        while True:
            try:
                if default:
                    user_input = input(f"{prompt} (default: {default}): ").strip()
                    if not user_input:
                        return default
                else:
                    user_input = input(f"{prompt}: ").strip()
                
                value = int(user_input)
                
                if value < min_val:
                    print(f"❌ Value must be at least {min_val}")
                    continue
                
                if max_val and value > max_val:
                    print(f"❌ Value must be at most {max_val}")
                    continue
                
                return value
            
            except ValueError:
                print("❌ Please enter a valid number")
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                sys.exit(0)
    
    def select_model_size(self) -> ModelSize:
        """Model size selection menu"""
        models = [
            ("7B Parameter Model", ModelSize.SMALL_7B),
            ("13B Parameter Model", ModelSize.MEDIUM_13B),
            ("GPT-4 Class Model", ModelSize.LARGE_GPT4)
        ]
        
        self.print_menu(
            "Model Size Selection",
            [f"{name} - {size.value}" for name, size in models]
        )
        
        choice = self.get_user_choice(len(models))
        
        if choice == 0:
            sys.exit(0)
        elif choice == len(models) + 1:
            return None  # Back
        else:
            return models[choice - 1][1]
    
    def select_hardware(self) -> HardwareType:
        """Hardware selection menu"""
        hardware_list = [
            ("RTX 4090", HardwareType.RTX_4090, "24GB VRAM, Consumer GPU"),
            ("A10G", HardwareType.A10G, "24GB VRAM, Cloud GPU"),
            ("A100 40GB", HardwareType.A100_40GB, "40GB HBM, Enterprise GPU"),
            ("A100 80GB", HardwareType.A100_80GB, "80GB HBM, Enterprise GPU"),
            ("H100", HardwareType.H100, "80GB HBM2e, Latest GPU"),
            ("CPU Only", HardwareType.CPU_ONLY, "CPU inference (slow)")
        ]
        
        self.print_menu(
            "Hardware Selection",
            [f"{name} - {desc}" for name, _, desc in hardware_list]
        )
        
        choice = self.get_user_choice(len(hardware_list))
        
        if choice == 0:
            sys.exit(0)
        elif choice == len(hardware_list) + 1:
            return None  # Back
        else:
            return hardware_list[choice - 1][1]
    
    def select_deployment_mode(self) -> DeploymentMode:
        """Deployment mode selection menu"""
        deployments = [
            ("Single GPU", DeploymentMode.SINGLE_GPU, "Single GPU deployment"),
            ("Multi-GPU", DeploymentMode.MULTI_GPU, "Multiple GPUs (2x cost)"),
            ("Distributed", DeploymentMode.DISTRIBUTED, "Distributed setup (4x cost)"),
            ("Cloud API", DeploymentMode.CLOUD_API, "Cloud API service")
        ]
        
        self.print_menu(
            "Deployment Mode",
            [f"{name} - {desc}" for name, _, desc in deployments]
        )
        
        choice = self.get_user_choice(len(deployments))
        
        if choice == 0:
            sys.exit(0)
        elif choice == len(deployments) + 1:
            return None  # Back
        else:
            return deployments[choice - 1][1]
    
    def select_precision(self) -> str:
        """Precision selection menu"""
        precisions = [
            ("FP16", "fp16", "Full precision, best quality"),
            ("INT8", "int8", "2x memory reduction, minimal quality loss"),
            ("INT4", "int4", "4x memory reduction, some quality loss")
        ]
        
        self.print_menu(
            "Precision Selection",
            [f"{name} - {desc}" for name, _, desc in precisions]
        )
        
        choice = self.get_user_choice(len(precisions))
        
        if choice == 0:
            sys.exit(0)
        elif choice == len(precisions) + 1:
            return None  # Back
        else:
            return precisions[choice - 1][1]
    
    def get_token_configuration(self) -> Tuple[int, int, int]:
        """Get token configuration from user"""
        print("\n🔤 Token Configuration")
        print("-" * 20)
        
        input_tokens = self.get_number_input(
            "Input tokens (prompt + context)", 
            min_val=1, 
            max_val=10000, 
            default=150
        )
        
        output_tokens = self.get_number_input(
            "Output tokens (response)", 
            min_val=1, 
            max_val=5000, 
            default=100
        )
        
        batch_size = self.get_number_input(
            "Batch size (requests processed together)", 
            min_val=1, 
            max_val=32, 
            default=1
        )
        
        return input_tokens, output_tokens, batch_size
    
    def display_results(self, result: InferenceResult, config: Dict):
        """Display calculation results with nice formatting"""
        print("\n" + "=" * 60)
        print("📊 CALCULATION RESULTS")
        print("=" * 60)
        
        # Configuration summary
        print("\n🔧 Configuration:")
        print(f"  Model: {config['model_size'].value}")
        print(f"  Hardware: {config['hardware_type'].value}")
        print(f"  Deployment: {config['deployment_mode'].value}")
        print(f"  Precision: {config['precision'].upper()}")
        print(f"  Tokens: {config['input_tokens']} input + {config['output_tokens']} output")
        print(f"  Batch Size: {config['batch_size']}")
        
        # Key metrics
        print("\n💾 Memory Usage:")
        print(f"  Total: {result.memory_usage_gb:.2f} GB")
        print(f"  Utilization: {result.memory_utilization:.1%}")
        
        print("\n⏱️ Performance:")
        print(f"  Latency: {result.latency_ms:.1f} ms")
        print(f"  Throughput: {result.tokens_per_second:.1f} tokens/second")
        
        print("\n💰 Cost Analysis:")
        print(f"  Per Request: ${result.cost_per_request:.6f}")
        print(f"  Per Token: ${result.cost_per_token:.8f}")
        
        # Cost projections
        requests_per_day = self.get_number_input(
            "\nEnter requests per day for cost projection",
            min_val=1,
            default=1000
        )
        
        daily_cost = result.cost_per_request * requests_per_day
        monthly_cost = daily_cost * 30
        yearly_cost = daily_cost * 365
        
        print(f"\n💸 Cost Projections ({requests_per_day} requests/day):")
        print(f"  Daily: ${daily_cost:.2f}")
        print(f"  Monthly: ${monthly_cost:.2f}")
        print(f"  Yearly: ${yearly_cost:.2f}")
        
        # Compatibility
        print(f"\n🔧 Hardware Compatibility: {'✅ Compatible' if result.hardware_compatible else '❌ Incompatible'}")
        
        # Recommendations
        print("\n💡 Recommendations:")
        for rec in result.recommendations:
            print(f"  {rec}")
    
    def hardware_comparison(self, config: Dict):
        """Compare performance across different hardware"""
        print("\n🔍 Hardware Comparison")
        print("-" * 30)
        
        hardware_types = [
            HardwareType.RTX_4090,
            HardwareType.A10G,
            HardwareType.A100_40GB,
            HardwareType.A100_80GB,
            HardwareType.H100
        ]
        
        results = []
        for hw_type in hardware_types:
            try:
                result = self.calculator.calculate_inference(
                    model_size=config['model_size'],
                    input_tokens=config['input_tokens'],
                    output_tokens=config['output_tokens'],
                    batch_size=config['batch_size'],
                    hardware_type=hw_type,
                    deployment_mode=config['deployment_mode'],
                    precision=config['precision']
                )
                results.append((hw_type, result))
            except Exception as e:
                print(f"⚠️ Could not calculate for {hw_type.value}: {str(e)}")
        
        if results:
            print(f"\n{'Hardware':<15} {'Memory':<10} {'Latency':<12} {'Cost/Req':<12} {'Tokens/s':<10} {'Compatible'}")
            print("-" * 75)
            
            for hw_type, result in results:
                compatible = "✅" if result.hardware_compatible else "❌"
                print(f"{hw_type.value:<15} {result.memory_usage_gb:<9.1f}G "
                      f"{result.latency_ms:<11.1f}ms ${result.cost_per_request:<11.6f} "
                      f"{result.tokens_per_second:<9.1f} {compatible}")
    
    def main_menu(self):
        """Main application menu"""
        while True:
            self.clear()
            self.print_header()
            
            self.print_menu(
                "Main Menu",
                [
                    "🧮 Calculate Inference Metrics",
                    "🔍 Hardware Comparison",
                    "📖 Quick Guide",
                    "ℹ️ About"
                ],
                include_back=False
            )
            
            choice = self.get_user_choice(4, include_back=False)
            
            if choice == 0:
                print("\n👋 Thank you for using LLM Inference Calculator!")
                sys.exit(0)
            elif choice == 1:
                self.calculate_workflow()
            elif choice == 2:
                self.hardware_comparison_workflow()
            elif choice == 3:
                self.show_quick_guide()
            elif choice == 4:
                self.show_about()
    
    def calculate_workflow(self):
        """Main calculation workflow"""
        self.clear()
        self.print_header()
        
        print("🧮 Inference Calculation Wizard")
        print("Follow the steps to configure your calculation")
        print()
        
        # Step 1: Model selection
        model_size = self.select_model_size()
        if model_size is None:
            return
        
        # Step 2: Hardware selection
        hardware_type = self.select_hardware()
        if hardware_type is None:
            return
        
        # Step 3: Deployment mode
        deployment_mode = self.select_deployment_mode()
        if deployment_mode is None:
            return
        
        # Step 4: Precision
        precision = self.select_precision()
        if precision is None:
            return
        
        # Step 5: Token configuration
        input_tokens, output_tokens, batch_size = self.get_token_configuration()
        
        # Perform calculation
        print("\n⏳ Calculating inference metrics...")
        
        config = {
            'model_size': model_size,
            'hardware_type': hardware_type,
            'deployment_mode': deployment_mode,
            'precision': precision,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'batch_size': batch_size
        }
        
        try:
            result = self.calculator.calculate_inference(
                model_size=model_size,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                batch_size=batch_size,
                hardware_type=hardware_type,
                deployment_mode=deployment_mode,
                precision=precision
            )
            
            self.display_results(result, config)
            
        except Exception as e:
            print(f"❌ Calculation failed: {str(e)}")
        
        input("\n📍 Press Enter to continue...")
    
    def hardware_comparison_workflow(self):
        """Hardware comparison workflow"""
        self.clear()
        self.print_header()
        
        print("🔍 Hardware Comparison")
        print("Compare different hardware options for your use case")
        print()
        
        # Get configuration (excluding hardware)
        model_size = self.select_model_size()
        if model_size is None:
            return
        
        deployment_mode = self.select_deployment_mode()
        if deployment_mode is None:
            return
        
        precision = self.select_precision()
        if precision is None:
            return
        
        input_tokens, output_tokens, batch_size = self.get_token_configuration()
        
        config = {
            'model_size': model_size,
            'deployment_mode': deployment_mode,
            'precision': precision,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'batch_size': batch_size
        }
        
        print("\n⏳ Comparing hardware options...")
        self.hardware_comparison(config)
        
        input("\n📍 Press Enter to continue...")
    
    def show_quick_guide(self):
        """Show quick usage guide"""
        self.clear()
        self.print_header()
        
        print("📖 Quick Guide")
        print("=" * 20)
        print()
        print("🎯 Purpose:")
        print("  This calculator helps estimate memory usage, latency, and costs")
        print("  for LLM inference deployments.")
        print()
        print("📊 Key Metrics:")
        print("  • Memory Usage: Model weights + KV cache + overhead")
        print("  • Latency: Time to process input and generate output")
        print("  • Cost: Based on hardware pricing and inference time")
        print("  • Throughput: Tokens generated per second")
        print()
        print("🔧 Model Sizes:")
        print("  • 7B: Good for most applications, cost-effective")
        print("  • 13B: Better quality, moderate resource requirements")
        print("  • GPT-4: Best quality, high resource requirements")
        print()
        print("💾 Precision Options:")
        print("  • FP16: Full precision, best quality")
        print("  • INT8: 2x memory reduction, minimal quality loss")
        print("  • INT4: 4x memory reduction, some quality loss")
        print()
        print("🚀 Tips:")
        print("  • Start with 7B models and scale up as needed")
        print("  • Use INT8 quantization for better efficiency")
        print("  • Consider cloud APIs for occasional high-quality needs")
        print("  • Batch processing can improve cost efficiency")
        
        input("\n📍 Press Enter to continue...")
    
    def show_about(self):
        """Show about information"""
        self.clear()
        self.print_header()
        
        print("ℹ️ About LLM Inference Calculator")
        print("=" * 35)
        print()
        print("Version: 1.0")
        print("Author: AI Assistant")
        print("Purpose: Help make informed decisions about LLM deployments")
        print()
        print("🎯 Features:")
        print("  • Memory usage calculation")
        print("  • Latency estimation")
        print("  • Cost analysis")
        print("  • Hardware compatibility checking")
        print("  • Optimization recommendations")
        print("  • Hardware comparison")
        print()
        print("⚠️ Disclaimer:")
        print("  Results are estimates based on typical configurations.")
        print("  Actual performance may vary depending on:")
        print("  • Specific model architecture")
        print("  • Hardware configuration")
        print("  • Software optimizations")
        print("  • Workload characteristics")
        print()
        print("📧 For questions or improvements, consult the documentation.")
        
        input("\n📍 Press Enter to continue...")
    
    def run(self):
        """Run the CLI application"""
        try:
            self.main_menu()
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            sys.exit(0)


def main():
    """Entry point for the CLI application"""
    cli = CalculatorCLI()
    cli.run()


if __name__ == "__main__":
    main() 
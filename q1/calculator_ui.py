#!/usr/bin/env python3
"""
LLM Inference Calculator - Web UI
Modern web interface for calculating LLM inference metrics
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from inference_calculator import (
    LLMInferenceCalculator, 
    ModelSize, 
    HardwareType, 
    DeploymentMode
)

# Page configuration
st.set_page_config(
    page_title="LLM Inference Calculator",
    page_icon="🧮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    font-weight: bold;
    text-align: center;
    color: #1f77b4;
    margin-bottom: 2rem;
}
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #1f77b4;
    margin: 0.5rem 0;
}
.recommendation-box {
    background-color: #e8f4fd;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #0066cc;
    margin: 0.5rem 0;
}
.warning-box {
    background-color: #fff3cd;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #ffc107;
    margin: 0.5rem 0;
}
.success-box {
    background-color: #d4edda;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #28a745;
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_calculator():
    """Initialize and cache the calculator instance"""
    return LLMInferenceCalculator()

def create_comparison_chart(results_df):
    """Create a comparison chart for different configurations"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Memory Usage (GB)', 'Latency (ms)', 'Cost per Request ($)', 'Tokens/Second'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Memory Usage
    fig.add_trace(
        go.Bar(x=results_df['Configuration'], y=results_df['Memory (GB)'], 
               name='Memory', marker_color='lightblue'),
        row=1, col=1
    )
    
    # Latency
    fig.add_trace(
        go.Bar(x=results_df['Configuration'], y=results_df['Latency (ms)'], 
               name='Latency', marker_color='lightcoral'),
        row=1, col=2
    )
    
    # Cost
    fig.add_trace(
        go.Bar(x=results_df['Configuration'], y=results_df['Cost per Request'], 
               name='Cost', marker_color='lightgreen'),
        row=2, col=1
    )
    
    # Tokens per second
    fig.add_trace(
        go.Bar(x=results_df['Configuration'], y=results_df['Tokens/Second'], 
               name='Throughput', marker_color='gold'),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=False, title_text="Performance Comparison")
    return fig

def create_cost_projection_chart(cost_per_request, requests_per_day):
    """Create cost projection over time"""
    days = np.arange(1, 31)
    daily_costs = cost_per_request * requests_per_day
    cumulative_costs = daily_costs * days
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=days, y=cumulative_costs,
        mode='lines+markers',
        name='Cumulative Cost',
        line=dict(color='#1f77b4', width=3)
    ))
    
    fig.update_layout(
        title='Monthly Cost Projection',
        xaxis_title='Days',
        yaxis_title='Cumulative Cost ($)',
        height=400
    )
    return fig

def format_recommendations(recommendations):
    """Format recommendations with proper styling"""
    formatted = []
    for rec in recommendations:
        if '⚠️' in rec:
            formatted.append(f'<div class="warning-box">{rec}</div>')
        elif '✅' in rec:
            formatted.append(f'<div class="success-box">{rec}</div>')
        else:
            formatted.append(f'<div class="recommendation-box">{rec}</div>')
    return formatted

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🧮 LLM Inference Calculator</h1>', unsafe_allow_html=True)
    st.markdown("**Calculate memory, latency, and costs for LLM inference deployments**")
    
    # Initialize calculator
    calculator = get_calculator()
    
    # Sidebar for input parameters
    st.sidebar.header("Configuration Parameters")
    
    # Model selection
    model_options = {
        "7B Parameter Model": ModelSize.SMALL_7B,
        "13B Parameter Model": ModelSize.MEDIUM_13B,
        "GPT-4 Class Model": ModelSize.LARGE_GPT4
    }
    selected_model = st.sidebar.selectbox(
        "Model Size",
        options=list(model_options.keys()),
        help="Choose the model size for inference"
    )
    model_size = model_options[selected_model]
    
    # Hardware selection
    hardware_options = {
        "RTX 4090": HardwareType.RTX_4090,
        "A10G": HardwareType.A10G,
        "A100 40GB": HardwareType.A100_40GB,
        "A100 80GB": HardwareType.A100_80GB,
        "H100": HardwareType.H100,
        "CPU Only": HardwareType.CPU_ONLY
    }
    selected_hardware = st.sidebar.selectbox(
        "Hardware Type",
        options=list(hardware_options.keys()),
        index=2,  # Default to A100 40GB
        help="Select the hardware configuration"
    )
    hardware_type = hardware_options[selected_hardware]
    
    # Deployment mode
    deployment_options = {
        "Single GPU": DeploymentMode.SINGLE_GPU,
        "Multi-GPU": DeploymentMode.MULTI_GPU,
        "Distributed": DeploymentMode.DISTRIBUTED,
        "Cloud API": DeploymentMode.CLOUD_API
    }
    selected_deployment = st.sidebar.selectbox(
        "Deployment Mode",
        options=list(deployment_options.keys()),
        help="Choose the deployment configuration"
    )
    deployment_mode = deployment_options[selected_deployment]
    
    # Token configuration
    st.sidebar.subheader("Token Configuration")
    input_tokens = st.sidebar.number_input(
        "Input Tokens", 
        min_value=1, 
        max_value=10000, 
        value=150,
        help="Number of input tokens (prompt + context)"
    )
    
    output_tokens = st.sidebar.number_input(
        "Output Tokens", 
        min_value=1, 
        max_value=5000, 
        value=100,
        help="Number of output tokens (response)"
    )
    
    batch_size = st.sidebar.number_input(
        "Batch Size", 
        min_value=1, 
        max_value=32, 
        value=1,
        help="Number of requests processed together"
    )
    
    # Precision selection
    precision = st.sidebar.selectbox(
        "Precision",
        options=["fp16", "int8", "int4"],
        help="Model precision (lower = faster, less memory, potentially lower quality)"
    )
    
    # Advanced settings
    with st.sidebar.expander("Advanced Settings"):
        requests_per_day = st.number_input(
            "Requests per Day", 
            min_value=1, 
            max_value=100000, 
            value=1000,
            help="For cost projections"
        )
        
        show_comparison = st.checkbox("Show Hardware Comparison", value=False)
    
    # Calculate button
    if st.sidebar.button("Calculate", type="primary"):
        st.session_state.calculate = True
    
    # Perform calculation
    if hasattr(st.session_state, 'calculate') and st.session_state.calculate:
        with st.spinner("Calculating inference metrics..."):
            result = calculator.calculate_inference(
                model_size=model_size,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                batch_size=batch_size,
                hardware_type=hardware_type,
                deployment_mode=deployment_mode,
                precision=precision
            )
        
        # Results section
        st.header("📊 Results")
        
        # Key metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="💾 Memory Usage",
                value=f"{result.memory_usage_gb:.2f} GB",
                delta=f"{result.memory_utilization:.1%} utilization"
            )
        
        with col2:
            st.metric(
                label="⏱️ Latency",
                value=f"{result.latency_ms:.1f} ms",
                delta=f"{result.tokens_per_second:.1f} tokens/sec"
            )
        
        with col3:
            st.metric(
                label="💰 Cost per Request",
                value=f"${result.cost_per_request:.6f}",
                delta=f"${result.cost_per_token:.8f} per token"
            )
        
        with col4:
            compatibility_color = "normal" if result.hardware_compatible else "inverse"
            st.metric(
                label="🔧 Compatibility",
                value="✅ Compatible" if result.hardware_compatible else "❌ Incompatible",
                delta=None
            )
        
        # Cost projections
        st.subheader("💸 Cost Projections")
        
        col1, col2 = st.columns(2)
        
        with col1:
            daily_cost = result.cost_per_request * requests_per_day
            monthly_cost = daily_cost * 30
            yearly_cost = daily_cost * 365
            
            st.write("**Cost Breakdown:**")
            st.write(f"• Daily: ${daily_cost:.2f}")
            st.write(f"• Monthly: ${monthly_cost:.2f}")
            st.write(f"• Yearly: ${yearly_cost:.2f}")
        
        with col2:
            # Cost projection chart
            fig_cost = create_cost_projection_chart(result.cost_per_request, requests_per_day)
            st.plotly_chart(fig_cost, use_container_width=True)
        
        # Recommendations
        st.subheader("💡 Recommendations")
        formatted_recs = format_recommendations(result.recommendations)
        for rec in formatted_recs:
            st.markdown(rec, unsafe_allow_html=True)
        
        # Hardware comparison (if enabled)
        if show_comparison:
            st.subheader("🔍 Hardware Comparison")
            
            comparison_results = []
            for hw_name, hw_type in hardware_options.items():
                try:
                    comp_result = calculator.calculate_inference(
                        model_size=model_size,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        batch_size=batch_size,
                        hardware_type=hw_type,
                        deployment_mode=deployment_mode,
                        precision=precision
                    )
                    
                    comparison_results.append({
                        'Configuration': hw_name,
                        'Memory (GB)': comp_result.memory_usage_gb,
                        'Latency (ms)': comp_result.latency_ms,
                        'Cost per Request': comp_result.cost_per_request,
                        'Tokens/Second': comp_result.tokens_per_second,
                        'Compatible': '✅' if comp_result.hardware_compatible else '❌'
                    })
                except Exception as e:
                    st.warning(f"Could not calculate for {hw_name}: {str(e)}")
            
            if comparison_results:
                results_df = pd.DataFrame(comparison_results)
                
                # Display table
                st.dataframe(results_df, use_container_width=True)
                
                # Display charts
                if len(results_df) > 1:
                    fig_comparison = create_comparison_chart(results_df)
                    st.plotly_chart(fig_comparison, use_container_width=True)
    
    # Information panel
    with st.expander("ℹ️ About This Calculator"):
        st.markdown("""
        This calculator estimates key metrics for LLM inference deployments:
        
        **Metrics Calculated:**
        - **Memory Usage**: Model weights + KV cache + overhead
        - **Latency**: Prefill time + decode time for token generation
        - **Cost**: Based on hardware pricing and inference time
        - **Throughput**: Tokens generated per second
        
        **Precision Options:**
        - **FP16**: Full precision, best quality
        - **INT8**: Quantized, 2x memory reduction, minimal quality loss
        - **INT4**: Aggressive quantization, 4x memory reduction
        
        **Hardware Compatibility:**
        The calculator checks if your chosen hardware has sufficient memory
        and provides optimization recommendations.
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("Built with ❤️ using Streamlit | LLM Inference Calculator v1.0")

if __name__ == "__main__":
    main() 
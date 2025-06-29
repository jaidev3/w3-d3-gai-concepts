"""app.py
Streamlit app providing a web interface for the Adaptive Prompt Optimizer.
Run with:  streamlit run q3/app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

import streamlit as st

# Ensure the directory containing this file (the `q3` folder) is on PYTHONPATH
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.append(str(APP_DIR))

# Local imports (work whether app is executed from project root or inside `q3`)
from optimizers.prompt_optimizer import optimize_prompt  # type: ignore
from optimizers.tool_strategies import SUPPORTED_TOOLS, TOOL_STRATEGIES  # type: ignore

st.set_page_config(page_title="Adaptive Prompt Optimizer", page_icon="🤖", layout="wide")

st.title("🤖 Adaptive Prompt Optimizer")
st.markdown("Transform your prompts to maximize effectiveness with different AI coding tools.")

# Example prompts for users to try
EXAMPLE_PROMPTS = {
    "Python Function": "Write a function to calculate the factorial of a number",
    "Web API": "Create a REST API endpoint for user authentication",
    "Data Processing": "Process a CSV file and generate summary statistics",
    "Algorithm": "Implement a binary search algorithm",
    "Database Query": "Write SQL to find the top 5 customers by revenue",
    "React Component": "Create a responsive navigation bar component"
}

with st.sidebar:
    st.header("⚙️ Settings")
    selected_tool = st.selectbox("🎯 Target AI Coding Tool", SUPPORTED_TOOLS, index=0)
    temperature = st.slider("🎨 Creativity (temperature)", 0.0, 1.0, 0.7, 0.05)
    
    st.markdown("---")
    st.header("💡 Example Prompts")
    selected_example = st.selectbox("Try an example:", ["Select an example..."] + list(EXAMPLE_PROMPTS.keys()))
    
    if selected_example != "Select an example...":
        if st.button("Load Example"):
            st.session_state.example_prompt = EXAMPLE_PROMPTS[selected_example]
    
    st.markdown("---")
    st.markdown(
        "<small>💡 <strong>Tip:</strong> OpenAI model controlled via <code>PROMPT_OPTIMIZER_MODEL</code> env variable.<br>"
        "Make sure you have set <code>OPENAI_API_KEY</code> before running.</small>",
        unsafe_allow_html=True,
    )

st.subheader("1️⃣ Enter your base prompt")

# Load example if selected
initial_value = ""
if hasattr(st.session_state, 'example_prompt'):
    initial_value = st.session_state.example_prompt
    del st.session_state.example_prompt

base_prompt = st.text_area(
    "Prompt", 
    value=initial_value,
    height=200, 
    placeholder="Describe the coding task or question you would normally ask the tool…",
    help="Enter the prompt you would normally use. The optimizer will tailor it for your selected AI tool."
)

# Add validation and tool-specific guidance
if base_prompt.strip():
    st.info(f"🎯 **Optimizing for {selected_tool}**: {TOOL_STRATEGIES[selected_tool][:100]}...")

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    optimize_clicked = st.button("🚀 Optimize Prompt", type="primary", disabled=not base_prompt.strip(), use_container_width=True)

if optimize_clicked:
    with st.spinner(f"🤖 Generating optimized prompt for {selected_tool}..."):
        try:
            result: Dict[str, Any] = optimize_prompt(base_prompt, selected_tool, temperature=temperature)
        except Exception as e:
            st.error(f"❌ Error while optimizing prompt: {e}")
            st.error("Make sure your OPENAI_API_KEY is set correctly.")
            st.stop()

    optimized_prompt = result.get("optimized_prompt") or "(Model did not return an optimized prompt.)"
    explanation = result.get("explanation") or "(No explanation provided.)"

    st.success("✅ Optimization complete!")
    
    # Before/After Comparison
    st.subheader("📊 Before vs After Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🔴 Original Prompt**")
        st.text_area("Original", value=base_prompt, height=150, disabled=True, key="original_display")
        
        # Analysis of original prompt
        original_word_count = len(base_prompt.split())
        original_char_count = len(base_prompt)
        st.caption(f"📝 {original_word_count} words • {original_char_count} characters")
    
    with col2:
        st.markdown("**🟢 Optimized Prompt**")
        st.text_area("Optimized", value=optimized_prompt, height=150, disabled=True, key="optimized_display")
        
        # Analysis of optimized prompt
        optimized_word_count = len(optimized_prompt.split())
        optimized_char_count = len(optimized_prompt)
        word_change = optimized_word_count - original_word_count
        char_change = optimized_char_count - original_char_count
        
        word_emoji = "📈" if word_change > 0 else "📉" if word_change < 0 else "➡️"
        char_emoji = "📈" if char_change > 0 else "📉" if char_change < 0 else "➡️"
        
        st.caption(f"📝 {optimized_word_count} words ({word_emoji} {word_change:+d}) • {optimized_char_count} characters ({char_emoji} {char_change:+d})")

    # Copy to clipboard functionality
    st.subheader("📋 Copy Optimized Prompt")
    st.code(optimized_prompt, language="text")
    
    # Explanation section
    st.subheader("💡 Optimization Explanation")
    st.markdown(explanation)
    
    # Additional insights
    with st.expander("🔍 Detailed Analysis"):
        st.markdown(f"**Target Tool:** {selected_tool}")
        st.markdown(f"**Temperature Used:** {temperature}")
        st.markdown(f"**Optimization Strategy Applied:**")
        st.markdown(TOOL_STRATEGIES[selected_tool])

    # Optional: persist analysis to local JSON for inspection
    save_path = Path(__file__).parent / "tool_analysis.json"
    entry = {
        "tool": selected_tool,
        "base_prompt": base_prompt,
        "optimized_prompt": optimized_prompt,
        "explanation": explanation,
    }
    try:
        if save_path.exists():
            data = json.loads(save_path.read_text())
            if not isinstance(data, list):
                data = []
        else:
            data = []
        data.append(entry)
        save_path.write_text(json.dumps(data, indent=2))
    except Exception:
        # Do not crash the app if we cannot write
        pass

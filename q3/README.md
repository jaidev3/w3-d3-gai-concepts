# 🤖 Adaptive Prompt Optimizer

This AI-powered tool helps you rewrite prompts so they are specifically tailored for different AI coding assistants. Maximize the effectiveness of your prompts by optimizing them for the unique strengths and capabilities of each tool.

## ✨ Features

### 🎯 **Multi-Tool Support**
- **GitHub Copilot**: Optimized for inline code completion and IDE integration
- **Cursor**: Enhanced for multi-step conversations and file operations
- **Replit**: Tailored for runnable REPL snippets and educational coding
- **AWS CodeWhisperer**: Specialized for AWS SDKs and cloud development
- **Tabnine**: Optimized for local context and pattern recognition
- **Sourcery**: Focused on Python refactoring and code quality

### 📊 **Advanced Analysis**
- **Before/After Comparison**: Side-by-side visual comparison of original vs optimized prompts
- **Word & Character Analysis**: Track changes in prompt length and complexity
- **Detailed Explanations**: Understand exactly what changes were made and why
- **Tool-Specific Strategies**: See the optimization approach used for each tool

### 🎨 **User-Friendly Interface**
- **Example Prompts**: Try pre-built examples for common coding tasks
- **Interactive Settings**: Adjust creativity levels and model parameters
- **Copy-to-Clipboard**: Easy copying of optimized prompts
- **Expandable Analysis**: Dive deep into optimization strategies

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Set your OpenAI API key
export OPENAI_API_KEY="your-api-key-here"

# (Optional) Choose a specific model
export PROMPT_OPTIMIZER_MODEL="gpt-4o-mini"
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the App
```bash
# From the project root
streamlit run q3/app.py
```

### 4. Start Optimizing
1. 📝 Enter your base prompt or select an example
2. 🎯 Choose your target AI coding tool
3. 🎨 Adjust creativity settings (optional)
4. 🚀 Click "Optimize Prompt"
5. 📊 Review the before/after comparison
6. 📋 Copy your optimized prompt

## 🔧 How It Works

### Architecture
- **Frontend**: Beautiful Streamlit web interface
- **Backend**: OpenAI GPT-powered optimization engine
- **Strategies**: Tool-specific optimization guidelines
- **Storage**: JSON-based analysis persistence

### Optimization Process
1. **Input Analysis**: Your prompt is analyzed for intent and complexity
2. **Strategy Selection**: Tool-specific optimization strategies are applied
3. **AI Enhancement**: OpenAI models enhance the prompt using expert guidance
4. **Result Parsing**: JSON response parsing for clean output
5. **Comparison**: Before/after analysis and metrics

### Tool-Specific Strategies
Each supported tool has carefully crafted optimization strategies:

- **GitHub Copilot**: Short, context-rich comments with type signatures
- **Cursor**: Multi-file operations with markdown code fences
- **Replit**: Runnable snippets with clear examples
- **AWS CodeWhisperer**: Service-specific with security best practices
- **Tabnine**: Pattern-based with local context emphasis
- **Sourcery**: Refactoring-focused with quality metrics

## 📁 Project Structure

```
q3/
├── app.py                    # Main Streamlit application
├── optimizers/              # Core optimization engine
│   ├── __init__.py          # Package initialization
│   ├── prompt_optimizer.py  # OpenAI integration & optimization logic
│   └── tool_strategies.py   # Tool-specific optimization strategies
├── tool_analysis.json      # Optimization history & analytics
└── README.md               # This file
```

## 🎯 Example Use Cases

### Python Function Development
**Original**: "Write a function to calculate factorial"
**Optimized for Copilot**: "# Calculate factorial of a number using recursion with type hints..."

### API Development
**Original**: "Create a user authentication endpoint"
**Optimized for Cursor**: "Create a secure REST API endpoint for user authentication with JWT tokens..."

### Data Processing
**Original**: "Process CSV data"
**Optimized for Replit**: "Process CSV file with pandas and generate summary statistics with example data..."

## 🔮 Future Enhancements

- **Additional AI Tools**: Support for more coding assistants
- **Custom Strategies**: User-defined optimization approaches
- **Batch Processing**: Optimize multiple prompts at once
- **Analytics Dashboard**: Usage patterns and effectiveness metrics
- **Template Library**: Pre-built prompt templates for common tasks

## 📊 Analytics & Persistence

The app automatically saves optimization results to `tool_analysis.json` for:
- **Performance Tracking**: Monitor optimization effectiveness
- **Strategy Analysis**: Understand which approaches work best
- **Usage Patterns**: Track most common optimization requests
- **Improvement Insights**: Data-driven strategy refinement

---

**Ready to supercharge your AI coding prompts?** 🚀 Launch the app and start optimizing!

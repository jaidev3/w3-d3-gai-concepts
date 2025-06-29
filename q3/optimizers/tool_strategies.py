"""tool_strategies.py
Registry of tool-specific optimization guidance for Adaptive Prompt Optimizer.
Feel free to extend or fine-tune the strategy text for better results.
"""

TOOL_STRATEGIES = {
    "GitHub Copilot": (
        "GitHub Copilot excels with inline code completion inside an IDE and works best with:\n"
        "• SHORT, context-rich comments placed immediately above the code to be generated\n"
        "• Imperative language (e.g. 'Create a function that...', 'Implement a class for...')\n"
        "• Type signatures, parameter descriptions, and return type hints\n"
        "• Edge cases and examples in the comment when possible\n"
        "• Specific function/variable names that indicate intent\n"
        "• Docstrings that describe expected behavior\n"
        "Avoid: Long explanatory text, multiple unrelated requests, or abstract concepts without concrete examples."
    ),
    "Cursor": (
        "Cursor supports multi-step conversation and can reference multiple files. Optimize for:\n"
        "• High-level refactors or file-generating tasks\n"
        "• File-path hints and desired file structure\n"
        "• Markdown code fences with explicit language specification\n"
        "• Multi-file context and cross-file references\n"
        "• Step-by-step implementation plans\n"
        "• Specific requirements about code style, patterns, or architecture\n"
        "• Instructions for testing and validation\n"
        "Leverage its ability to understand project structure and make coordinated changes across files."
    ),
    "Replit": (
        "Replit Chat focuses on runnable REPL snippets and educational coding. Optimize for:\n"
        "• Concise descriptions followed by exact code examples\n"
        "• Runnable, self-contained code snippets\n"
        "• Test cases or expected outputs that the snippet must satisfy\n"
        "• Learning-oriented explanations\n"
        "• Interactive examples that can be modified and tested\n"
        "• Clear input/output specifications\n"
        "Perfect for prototyping, learning, and quick experiments."
    ),
    "AWS CodeWhisperer": (
        "AWS CodeWhisperer is privacy-oriented and fine-tuned for AWS SDKs. Optimize for:\n"
        "• Specific AWS service names and use cases\n"
        "• Target programming language (Python, Java, JavaScript, etc.)\n"
        "• Relevant SDK classes, methods, or boto3 patterns\n"
        "• Infrastructure-as-code templates (CloudFormation, CDK)\n"
        "• Security best practices and IAM considerations\n"
        "• Cost optimization patterns\n"
        "• Error handling for AWS service calls\n"
        "Works best with enterprise-grade, production-ready AWS implementations."
    ),
    "Tabnine": (
        "Tabnine relies heavily on local code context and patterns. Optimize for:\n"
        "• Detailed docstrings and type hints near the completion point\n"
        "• Consistent naming conventions and code patterns\n"
        "• Clear variable and function names that indicate intent\n"
        "• Local context from surrounding code\n"
        "• Common programming patterns and idioms\n"
        "• Explicit type annotations for better suggestions\n"
        "• Comments that describe the intended algorithm or approach\n"
        "Works best when the codebase has consistent patterns and good documentation."
    ),
    "Sourcery": (
        "Sourcery specializes in Python refactoring and code quality improvements. Optimize for:\n"
        "• Clear refactor goals (performance, readability, maintainability)\n"
        "• Specific functions, classes, or modules to transform\n"
        "• Code quality metrics to optimize (complexity, duplication, etc.)\n"
        "• Python best practices and PEP compliance\n"
        "• Performance optimization opportunities\n"
        "• Modern Python idioms and patterns\n"
        "• Specific refactoring techniques (extract method, simplify conditionals, etc.)\n"
        "Focus on measurable improvements to existing Python code."
    ),
}

SUPPORTED_TOOLS = list(TOOL_STRATEGIES.keys()) 
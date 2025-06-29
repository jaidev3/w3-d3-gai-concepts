"""prompt_optimizer.py
Core logic for transforming a base prompt into a tool-specific optimized prompt.
"""
from __future__ import annotations

import json
import os
from typing import Dict, Any

import openai
from dotenv import load_dotenv

load_dotenv()

from .tool_strategies import TOOL_STRATEGIES

# import openai api key
openai.api_key = os.getenv("OPENAI_API_KEY")

OPENAI_MODEL = os.getenv("PROMPT_OPTIMIZER_MODEL", "gpt-3.5-turbo-16k")


def optimize_prompt(
    base_prompt: str,
    tool: str,
    model: str | None = None,
    temperature: float = 0.7,
) -> Dict[str, Any]:
    """Return an optimized prompt and the explanation.

    Parameters
    ----------
    base_prompt: str
        The original prompt provided by the user.
    tool: str
        The target AI coding tool (must exist inside TOOL_STRATEGIES).
    model: str | None
        Override the OpenAI model name.
    temperature: float
        The temperature for the OpenAI model.

    Returns
    -------
    Dict[str, Any]
        {"optimized_prompt": str, "explanation": str}
    """

    if tool not in TOOL_STRATEGIES:
        raise ValueError(f"Unknown tool '{tool}'. Supported tools: {list(TOOL_STRATEGIES)}")

    model_name = model or OPENAI_MODEL

    system_instruction = (
        f"You are an elite Prompt Engineer. "
        f"Given a base prompt, you must rewrite it specifically for the {tool} AI coding assistant. "
        f"The objective is to maximise {tool}'s strengths while mitigating its weaknesses, according to the following guidance:\n"
        f"{TOOL_STRATEGIES[tool]}\n"
        "Return a JSON object with keys: optimized_prompt, explanation."
    )

    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": base_prompt.strip()},
    ]

    # Temperature controls creativity; keep default 0.7 unless overridden
    response = openai.chat.completions.create(
        model=model_name,
        messages=messages,  # type: ignore[arg-type]
        temperature=temperature,
    )

    # Extract assistant content (OpenAI v1 object has `.content` attribute)
    assistant_message = response.choices[0].message
    content = assistant_message.content
    if content is None:
        content = str(assistant_message)
    else:
        content = content.strip()

    # Attempt to parse JSON from assistant reply
    optimized_prompt: str | None = None
    explanation: str | None = None
    try:
        # Some models wrap JSON in markdown fences (```json\n{...}\n```)
        if content.startswith("```"):
            # Split on first and second triple backticks to isolate JSON
            parts = content.split("```")
            # Find first part that looks like JSON (contains '{' and '}')
            json_segment = next((p for p in parts if "{" in p and "}" in p), "{}")
            content = json_segment
        data = json.loads(content)
        optimized_prompt = data.get("optimized_prompt")
        explanation = data.get("explanation")
    except (json.JSONDecodeError, StopIteration):
        # Treat entire content as explanation when JSON not parsed
        explanation = content

    return {
        "optimized_prompt": optimized_prompt or "",
        "explanation": explanation or "",
    }

"""
LLM Integration Module
=====================

Provides LLM (Large Language Model) integration for the MCP service.
Supports Ollama/Llama models for reasoning and decision-making.
"""

from .llama_integration import (
    LlamaReasoningEngine,
    TradingContext,
    LlamaResponse
)

__all__ = [
    "LlamaReasoningEngine",
    "TradingContext",
    "LlamaResponse"
]


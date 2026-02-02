"""
MCP Service - Model Context Protocol Service
============================================

A standalone service for MCP (Model Context Protocol) operations including:
- LLM integration (Groq)
- Chat handling
- Trading agents
- Market analysis tools
- Server management
"""

__version__ = "1.0.0"
__author__ = "Trading Bot AI Team"

import logging

logger = logging.getLogger(__name__)

# Import main components
from .server.mcp_trading_server import MCPTradingServer, MCPToolResult, MCPToolStatus
from .llm import GroqReasoningEngine, TradingContext, GroqResponse
from .chat import ChatHandler, ChatMessage, ChatResponse

# Import agents
from .agents.trading_agent import TradingAgent
from .agents.explanation_agent import ExplanationAgent
from .agents.insight_agent import InsightAgent

__all__ = [
    "MCPTradingServer",
    "MCPToolResult",
    "MCPToolStatus",
    "GroqReasoningEngine",
    "TradingContext",
    "GroqResponse",
    "ChatHandler",
    "ChatMessage",
    "ChatResponse",
    "TradingAgent",
    "ExplanationAgent",
    "InsightAgent"
]

logger.info("MCP Service components loaded successfully")


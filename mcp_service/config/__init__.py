"""
Configuration Module for MCP Service
====================================

Provides configuration management for the MCP service.
"""

import os
from typing import Dict, Any

def get_default_config() -> Dict[str, Any]:
    """Get default configuration"""
    return {
        "mcp": {
            "monitoring_port": int(os.getenv("MCP_MONITORING_PORT", "8002")),
            "max_sessions": int(os.getenv("MCP_MAX_SESSIONS", "100"))
        },
        "llama": {
            "llama_base_url": os.getenv("LLAMA_BASE_URL", "http://localhost:11434"),
            "llama_model": os.getenv("LLAMA_MODEL", "llama3.1:8b"),
            "max_tokens": int(os.getenv("LLAMA_MAX_TOKENS", "2048")),
            "temperature": float(os.getenv("LLAMA_TEMPERATURE", "0.7"))
        },
        "chat": {
            "max_history": int(os.getenv("CHAT_MAX_HISTORY", "50"))
        },
        "trading_agent": {
            "enable_ml": os.getenv("ENABLE_ML", "true").lower() == "true"
        }
    }

__all__ = ["get_default_config"]


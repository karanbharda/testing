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
        "chat": {
            "max_history": int(os.getenv("CHAT_MAX_HISTORY", "50"))
        },
        "trading_agent": {
            "enable_ml": os.getenv("ENABLE_ML", "true").lower() == "true"
        }
    }

__all__ = ["get_default_config"]


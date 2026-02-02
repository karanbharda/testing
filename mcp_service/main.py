#!/usr/bin/env python3
"""
MCP Service Entry Point
=======================

Standalone entry point for the MCP (Model Context Protocol) service.
Run this file to start the MCP service independently.

Usage:
    python mcp_service/main.py
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "backend"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler('mcp_service.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

async def main():
    """Main entry point for MCP service"""
    try:
        logger.info("=" * 60)
        logger.info("Starting MCP Service")
        logger.info("=" * 60)
        
        # Import MCP components
        from mcp_service.server.mcp_trading_server import MCPTradingServer
        from mcp_service.llm import GroqReasoningEngine
        from mcp_service.chat import ChatHandler
        from mcp_service.agents import TradingAgent
        
        # Load configuration
        config = {
            "mcp": {
                "monitoring_port": int(os.getenv("MCP_MONITORING_PORT", "8002")),
                "max_sessions": int(os.getenv("MCP_MAX_SESSIONS", "100"))
            },
            "groq": {
                "groq_api_key": os.getenv("GROQ_API_KEY", ""),
                "groq_model": os.getenv("GROQ_MODEL", "llama3-8b-8192"),
                "max_tokens": int(os.getenv("GROQ_MAX_TOKENS", "2048")),
                "temperature": float(os.getenv("GROQ_TEMPERATURE", "0.7"))
            },
            "chat": {
                "max_history": int(os.getenv("CHAT_MAX_HISTORY", "50"))
            },
            "trading_agent": {
                "enable_ml": os.getenv("ENABLE_ML", "true").lower() == "true"
            }
        }
        
        # Initialize MCP server
        logger.info("Initializing MCP Trading Server...")
        mcp_server = MCPTradingServer(config["mcp"])
        
        # Initialize LLM engine
        logger.info("Initializing LLM Engine...")
        groq_engine = GroqReasoningEngine(config["groq"])
        
        # Initialize Chat Handler
        logger.info("Initializing Chat Handler...")
        chat_handler = ChatHandler({
            **config["chat"],
            "groq": config["groq"],
            "trading_agent": config["trading_agent"]
        })
        
        # Initialize Trading Agent
        logger.info("Initializing Trading Agent...")
        trading_agent = TradingAgent({
            **config["trading_agent"],
            "groq": config["groq"]
        })
        await trading_agent.initialize()
        
        logger.info("=" * 60)
        logger.info("MCP Service started successfully!")
        logger.info(f"Monitoring port: {config['mcp']['monitoring_port']}")
        logger.info(f"LLM Model: {config['groq']['groq_model']}")
        logger.info("=" * 60)
        
        # Health check
        health_status = mcp_server.get_health_status()
        logger.info(f"Server Health: {health_status}")
        
        # Keep service running
        logger.info("MCP Service is running. Press Ctrl+C to stop.")
        try:
            while True:
                await asyncio.sleep(60)
                # Periodic health check
                health_status = mcp_server.get_health_status()
                logger.debug(f"Health check: {health_status}")
        except KeyboardInterrupt:
            logger.info("Shutting down MCP Service...")
        
    except Exception as e:
        logger.error(f"Error starting MCP service: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())


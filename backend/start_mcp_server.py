#!/usr/bin/env python3
"""
MCP Server Startup Script
==========================

Comprehensive startup script for all MCP (Model Context Protocol) components:
- MCP Trading Server
- LLM Engine (Llama/Ollama)
- Chat Handler
- Trading Agents
- Market Analysis Tools
- API Server for chat requests
- All related services

Usage:
    python backend/start_mcp_server.py
"""

import asyncio
import logging
import os
import sys
import signal
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

# Add project root to path
# Script is in backend/, but mcp_service is at root level
project_root = Path(__file__).parent.parent  # Go up one level to project root
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "backend"))

# Fix Windows console encoding for Unicode characters
if sys.platform == 'win32':
    import io
    # Fix stdout/stderr encoding for emoji support
    if hasattr(sys.stdout, 'buffer'):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'buffer'):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Configure comprehensive logging
log_dir = project_root / "logs"
log_dir.mkdir(exist_ok=True)

# Configure logging with UTF-8 encoding
file_handler = logging.FileHandler(log_dir / 'mcp_server.log', encoding='utf-8')
console_handler = logging.StreamHandler(sys.stdout)  # Use sys.stdout which is now UTF-8 on Windows

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[file_handler, console_handler]
)
logger = logging.getLogger(__name__)

# Global references for graceful shutdown
mcp_server = None
groq_engine = None
chat_handler = None
trading_agent = None
all_agents = {}
all_tools = {}
api_task = None

def load_config() -> Dict[str, Any]:
    """Load configuration from environment variables"""
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
            "enable_ml": os.getenv("ENABLE_ML", "true").lower() == "true",
            "fyers": {
                "fyers_access_token": os.getenv("FYERS_ACCESS_TOKEN", ""),
                "fyers_client_id": os.getenv("FYERS_APP_ID", "")
            }
        },
        "fyers": {
            "fyers_access_token": os.getenv("FYERS_ACCESS_TOKEN", ""),
            "fyers_client_id": os.getenv("FYERS_APP_ID", "")
        }
    }
    return config

async def initialize_llm_engine(config: Dict[str, Any]) -> Any:
    """Initialize LLM Engine"""
    try:
        logger.info("=" * 60)
        logger.info("Initializing LLM Engine (Groq)...")
        logger.info("=" * 60)
        
        from mcp_service.llm import GroqReasoningEngine
        
        groq_config = config.get("groq", {})
        engine = GroqReasoningEngine(groq_config)
        
        # Health check
        health = await engine.health_check()
        if health.get("status") == "healthy":
            logger.info(f"[OK] LLM Engine initialized successfully")
            logger.info(f"   Model: {groq_config.get('groq_model')}")
            logger.info(f"   API Key: {'*' * len(groq_config.get('groq_api_key', ''))}")
        else:
            logger.warning(f"[WARN] LLM Engine health check failed: {health.get('error')}")
            logger.warning("   Continuing with limited functionality...")
        
        return engine
    except Exception as e:
        logger.error(f"[ERROR] Failed to initialize LLM Engine: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

async def initialize_chat_handler(config: Dict[str, Any], groq_engine: Any) -> Any:
    """Initialize Chat Handler"""
    try:
        logger.info("=" * 60)
        logger.info("Initializing Chat Handler...")
        logger.info("=" * 60)
        
        from mcp_service.chat import ChatHandler
        
        chat_config = {
            **config.get("chat", {}),
            "groq": config.get("groq", {}),
            "trading_agent": config.get("trading_agent", {})
        }
        handler = ChatHandler(chat_config)
        
        logger.info("[OK] Chat Handler initialized successfully")
        return handler
    except Exception as e:
        logger.error(f"[ERROR] Failed to initialize Chat Handler: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

async def initialize_trading_agents(config: Dict[str, Any]) -> Dict[str, Any]:
    """Initialize all trading agents"""
    agents = {}
    
    try:
        logger.info("=" * 60)
        logger.info("Initializing Trading Agents...")
        logger.info("=" * 60)
        
        from mcp_service.agents import (
            TradingAgent,
            ExplanationAgent,
            InsightAgent,
            RiskAgent,
            PortfolioAgent
        )
        
        # Trading Agent
        try:
            logger.info("  [->] Initializing Trading Agent...")
            trading_config = {
                **config.get("trading_agent", {}),
                "llama": config.get("llama", {}),
                "fyers": config.get("fyers", {})
            }
            trading_agent = TradingAgent(trading_config)
            await trading_agent.initialize()
            agents["trading"] = trading_agent
            logger.info("  [OK] Trading Agent initialized")
        except Exception as e:
            logger.error(f"  [ERROR] Trading Agent failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())
        
        # Explanation Agent
        try:
            logger.info("  [->] Initializing Explanation Agent...")
            explanation_config = {
                "llama": config.get("llama", {})
            }
            explanation_agent = ExplanationAgent(explanation_config)
            agents["explanation"] = explanation_agent
            logger.info("  [OK] Explanation Agent initialized")
        except Exception as e:
            logger.error(f"  [ERROR] Explanation Agent failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())
        
        # Insight Agent
        try:
            logger.info("  [->] Initializing Insight Agent...")
            insight_config = {
                "llama": config.get("llama", {})
            }
            insight_agent = InsightAgent(insight_config)
            agents["insight"] = insight_agent
            logger.info("  [OK] Insight Agent initialized")
        except Exception as e:
            logger.error(f"  [ERROR] Insight Agent failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())
        
        # Risk Agent
        try:
            logger.info("  [->] Initializing Risk Agent...")
            risk_agent = RiskAgent(config.get("trading_agent", {}))
            agents["risk"] = risk_agent
            logger.info("  [OK] Risk Agent initialized")
        except Exception as e:
            logger.error(f"  [ERROR] Risk Agent failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())
        
        # Portfolio Agent
        try:
            logger.info("  [->] Initializing Portfolio Agent...")
            portfolio_agent = PortfolioAgent(config.get("trading_agent", {}))
            agents["portfolio"] = portfolio_agent
            logger.info("  [OK] Portfolio Agent initialized")
        except Exception as e:
            logger.error(f"  [ERROR] Portfolio Agent failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())
        
        logger.info(f"[OK] Initialized {len(agents)} agents")
        return agents
        
    except Exception as e:
        logger.error(f"[ERROR] Failed to initialize agents: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return agents

async def initialize_tools(config: Dict[str, Any], agents: Dict[str, Any]) -> Dict[str, Any]:
    """Initialize all trading tools"""
    tools = {}
    
    try:
        logger.info("=" * 60)
        logger.info("Initializing Trading Tools...")
        logger.info("=" * 60)
        
        from mcp_service.tools import (
            ExecutionTool,
            MarketAnalysisTool,
            PortfolioTool,
            RiskManagementTool,
            SentimentTool,
            PredictionTool,
            ScanTool
        )
        
        tool_config = {
            **config.get("trading_agent", {}),
            "llama": config.get("llama", {}),
            "fyers": config.get("fyers", {})
        }
        
        # Initialize Fyers client if needed
        fyers_client = None
        try:
            from fyers_client import FyersAPIClient
            fyers_config = config.get("fyers", {})
            if fyers_config.get("fyers_access_token") and fyers_config.get("fyers_client_id"):
                fyers_client = FyersAPIClient(fyers_config)
                logger.info("  [OK] Fyers client initialized for tools")
        except Exception as e:
            logger.warning(f"  [WARN] Fyers client not available: {e}")
        
        # Market Analysis Tool
        try:
            logger.info("  [->] Initializing Market Analysis Tool...")
            if fyers_client:
                market_tool = MarketAnalysisTool(fyers_client)
            else:
                try:
                    market_tool = MarketAnalysisTool(tool_config)
                except:
                    market_tool = MarketAnalysisTool(None) if MarketAnalysisTool else None
            if market_tool:
                tools["market_analysis"] = market_tool
                logger.info("  [OK] Market Analysis Tool initialized")
            else:
                logger.warning("  [WARN] Market Analysis Tool could not be created")
        except Exception as e:
            logger.error(f"  [ERROR] Market Analysis Tool failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())
        
        # Portfolio Tool
        try:
            logger.info("  [->] Initializing Portfolio Tool...")
            portfolio_config = {
                **tool_config,
                "portfolio_agent": agents.get("portfolio"),
                "risk_agent": agents.get("risk")
            }
            portfolio_tool = PortfolioTool(portfolio_config)
            tools["portfolio"] = portfolio_tool
            logger.info("  [OK] Portfolio Tool initialized")
        except Exception as e:
            logger.error(f"  [ERROR] Portfolio Tool failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())
        
        # Risk Management Tool
        try:
            logger.info("  [->] Initializing Risk Management Tool...")
            risk_config = {
                **tool_config,
                "risk_agent": agents.get("risk")
            }
            risk_tool = RiskManagementTool(risk_config)
            tools["risk_management"] = risk_tool
            logger.info("  [OK] Risk Management Tool initialized")
        except Exception as e:
            logger.error(f"  [ERROR] Risk Management Tool failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())
        
        # Execution Tool
        try:
            logger.info("  [->] Initializing Execution Tool...")
            execution_config = {
                **tool_config,
                "fyers_client": fyers_client
            }
            execution_tool = ExecutionTool(execution_config)
            tools["execution"] = execution_tool
            logger.info("  [OK] Execution Tool initialized")
        except Exception as e:
            logger.error(f"  [ERROR] Execution Tool failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())
        
        # Sentiment Tool
        try:
            logger.info("  [->] Initializing Sentiment Tool...")
            sentiment_tool = SentimentTool(tool_config)
            tools["sentiment"] = sentiment_tool
            logger.info("  [OK] Sentiment Tool initialized")
        except Exception as e:
            logger.error(f"  [ERROR] Sentiment Tool failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())
        
        # Prediction Tool
        try:
            logger.info("  [->] Initializing Prediction Tool...")
            prediction_tool = PredictionTool(tool_config)
            tools["prediction"] = prediction_tool
            logger.info("  [OK] Prediction Tool initialized")
        except Exception as e:
            logger.error(f"  [ERROR] Prediction Tool failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())
        
        # Scan Tool
        try:
            logger.info("  [->] Initializing Scan Tool...")
            scan_config = {
                **tool_config,
                "fyers_client": fyers_client
            }
            scan_tool = ScanTool(scan_config)
            tools["scan"] = scan_tool
            logger.info("  [OK] Scan Tool initialized")
        except Exception as e:
            logger.error(f"  [ERROR] Scan Tool failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())
        
        logger.info(f"[OK] Initialized {len(tools)} tools")
        return tools
        
    except Exception as e:
        logger.error(f"[ERROR] Failed to initialize tools: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return tools

async def register_tools_with_server(server: Any, tools: Dict[str, Any]):
    """Register all tools with the MCP server"""
    try:
        logger.info("=" * 60)
        logger.info("Registering Tools with MCP Server...")
        logger.info("=" * 60)
        
        registered_count = 0
        
        # Tool registration mapping
        tool_methods = {
            "market_analysis": ("analyze_market", "Market Analysis Tool - Comprehensive market analysis"),
            "portfolio": ("analyze_portfolio", "Portfolio Tool - Portfolio analysis and optimization"),
            "risk_management": ("assess_risk", "Risk Management Tool - Risk assessment and management"),
            "execution": ("execute_trade", "Execution Tool - Trade execution"),
            "sentiment": ("analyze_sentiment", "Sentiment Tool - Market sentiment analysis"),
            "prediction": ("predict_price", "Prediction Tool - Price prediction"),
            "scan": ("scan_market", "Scan Tool - Market scanning")
        }
        
        for tool_name, tool_instance in tools.items():
            try:
                # Get method name and description
                method_name, description = tool_methods.get(tool_name, (None, f"{tool_name} tool"))
                
                if method_name and hasattr(tool_instance, method_name):
                    # Create wrapper function with proper closure
                    def make_wrapper(tool_inst, method):
                        async def wrapper(args, session_id):
                            return await getattr(tool_inst, method)(args, session_id)
                        return wrapper
                    
                    tool_function = make_wrapper(tool_instance, method_name)
                    
                    # Register tool
                    server.register_tool(
                        name=tool_name,
                        function=tool_function,
                        description=description,
                        schema={
                            "type": "object",
                            "properties": {},
                            "required": []
                        },
                        timeout=60
                    )
                    registered_count += 1
                    logger.info(f"  [OK] Registered: {tool_name}")
                else:
                    logger.warning(f"  [WARN] Tool {tool_name} doesn't have expected method {method_name}")
            except Exception as e:
                logger.error(f"  [ERROR] Failed to register {tool_name}: {e}")
                import traceback
                logger.debug(traceback.format_exc())
        
        logger.info(f"[OK] Registered {registered_count} tools with MCP server")
        
    except Exception as e:
        logger.error(f"[ERROR] Failed to register tools: {e}")
        import traceback
        logger.error(traceback.format_exc())

async def initialize_mcp_server(config: Dict[str, Any], tools: Dict[str, Any]) -> Any:
    """Initialize MCP Trading Server"""
    try:
        logger.info("=" * 60)
        logger.info("Initializing MCP Trading Server...")
        logger.info("=" * 60)
        
        from mcp_service.server.mcp_trading_server import MCPTradingServer
        
        mcp_config = config.get("mcp", {})
        server = MCPTradingServer(mcp_config)
        
        # Register tools
        await register_tools_with_server(server, tools)
        
        # Health check
        health_status = server.get_health_status()
        logger.info(f"[OK] MCP Server initialized successfully")
        logger.info(f"   Server ID: {health_status.get('server_id')}")
        logger.info(f"   Monitoring Port: {mcp_config.get('monitoring_port')}")
        logger.info(f"   Registered Tools: {health_status.get('registered_tools', 0)}")
        
        return server
        
    except Exception as e:
        logger.error(f"[ERROR] Failed to initialize MCP Server: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

async def start_mcp_api_server(chat_hdlr, llama_eng, trading_agt, config: Dict[str, Any]):
    """Start the MCP API server for handling chat requests"""
    try:
        logger.info("=" * 60)
        logger.info("Starting MCP API Server...")
        logger.info("=" * 60)
        
        from mcp_service.api_server import initialize_api, start_api_server
        
        # Initialize API with components
        initialize_api(chat_hdlr, trading_agt)
        
        # Get API server config
        api_host = os.getenv("MCP_API_HOST", "0.0.0.0")
        api_port = int(os.getenv("MCP_API_PORT", "8003"))
        
        # Start API server in background
        api_task = asyncio.create_task(
            start_api_server(host=api_host, port=api_port)
        )
        
        logger.info(f"[OK] MCP API Server started on {api_host}:{api_port}")
        logger.info(f"     Chat endpoint: http://{api_host}:{api_port}/api/chat")
        logger.info(f"     Health endpoint: http://{api_host}:{api_port}/api/health")
        
        return api_task
        
    except Exception as e:
        logger.error(f"[ERROR] Failed to start API server: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def setup_signal_handlers():
    """Setup signal handlers for graceful shutdown - returns event"""
    shutdown_event = asyncio.Event()
    
    def signal_handler(signum, frame):
        logger.info(f"\n[STOP] Received signal {signum}, shutting down gracefully...")
        shutdown_event.set()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    if hasattr(signal, 'SIGBREAK'):  # Windows
        signal.signal(signal.SIGBREAK, signal_handler)
    
    return shutdown_event

async def shutdown():
    """Graceful shutdown of all components"""
    global mcp_server, groq_engine, chat_handler, trading_agent, all_agents, all_tools, api_task
    
    logger.info("=" * 60)
    logger.info("Shutting down MCP Server components...")
    logger.info("=" * 60)
    
    try:
        # Cancel API server task
        if api_task:
            try:
                api_task.cancel()
                await api_task
            except asyncio.CancelledError:
                logger.info("[OK] API server task cancelled")
            except Exception as e:
                logger.error(f"[ERROR] Error cancelling API server: {e}")
        
        # Cleanup LLM engine
        if groq_engine:
            try:
                await groq_engine.cleanup()
                logger.info("[OK] LLM Engine cleaned up")
            except Exception as e:
                logger.error(f"[ERROR] Error cleaning up LLM Engine: {e}")
        
        # Cleanup agents
        for agent_name, agent in all_agents.items():
            try:
                if hasattr(agent, 'cleanup'):
                    await agent.cleanup()
                logger.info(f"[OK] {agent_name} agent cleaned up")
            except Exception as e:
                logger.error(f"[ERROR] Error cleaning up {agent_name} agent: {e}")
        
        # Cleanup tools
        for tool_name, tool in all_tools.items():
            try:
                if hasattr(tool, 'cleanup'):
                    await tool.cleanup()
                logger.info(f"[OK] {tool_name} tool cleaned up")
            except Exception as e:
                logger.error(f"[ERROR] Error cleaning up {tool_name} tool: {e}")
        
        logger.info("[OK] Shutdown complete")
        
    except Exception as e:
        logger.error(f"[ERROR] Error during shutdown: {e}")

async def main():
    """Main entry point - Initialize and run all MCP components"""
    global mcp_server, groq_engine, chat_handler, trading_agent, all_agents, all_tools, api_task
    
    try:
        logger.info("")
        logger.info("=" * 60)
        logger.info("[START] Starting MCP Server - All Components")
        logger.info("=" * 60)
        logger.info(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("")
        
        # Load configuration
        config = load_config()
        
        # Initialize components in order
        # 1. LLM Engine
        groq_engine = await initialize_llm_engine(config)
        
        # 2. Chat Handler
        chat_handler = await initialize_chat_handler(config, groq_engine)
        
        # 3. Trading Agents
        all_agents = await initialize_trading_agents(config)
        trading_agent = all_agents.get("trading")
        
        # 4. Trading Tools
        all_tools = await initialize_tools(config, all_agents)
        
        # 5. MCP Server (registers tools)
        mcp_server = await initialize_mcp_server(config, all_tools)
        
        # 6. Start API Server for chat requests
        api_task = await start_mcp_api_server(chat_handler, groq_engine, trading_agent, config)
        
        # Setup signal handlers
        shutdown_event = setup_signal_handlers()
        
        # Summary
        logger.info("")
        logger.info("=" * 60)
        logger.info("[OK] MCP Server Started Successfully!")
        logger.info("=" * 60)
        logger.info(f"Components Initialized:")
        logger.info(f"  - LLM Engine: {'[OK]' if groq_engine else '[FAIL]'}")
        logger.info(f"  - Chat Handler: {'[OK]' if chat_handler else '[FAIL]'}")
        logger.info(f"  - Trading Agents: {len(all_agents)}")
        logger.info(f"  - Trading Tools: {len(all_tools)}")
        logger.info(f"  - MCP Server: {'[OK]' if mcp_server else '[FAIL]'}")
        logger.info(f"  - API Server: {'[OK]' if api_task else '[FAIL]'}")
        logger.info("")
        logger.info("MCP Server is running. Press Ctrl+C to stop.")
        logger.info("=" * 60)
        logger.info("")
        
        # Keep service running
        try:
            while not shutdown_event.is_set():
                # Wait for either 60 seconds or shutdown event
                try:
                    await asyncio.wait_for(shutdown_event.wait(), timeout=60.0)
                    # Shutdown event was set
                    break
                except asyncio.TimeoutError:
                    # Timeout - do periodic health check
                    if mcp_server:
                        health_status = mcp_server.get_health_status()
                        logger.debug(f"Health check: {health_status.get('status')}")
        except KeyboardInterrupt:
            logger.info("\n[STOP] Keyboard interrupt received")
        
        # Shutdown gracefully
        await shutdown()
        
    except Exception as e:
        logger.error(f"[ERROR] Fatal error starting MCP server: {e}")
        import traceback
        logger.error(traceback.format_exc())
        await shutdown()
        sys.exit(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n[STOP] Shutting down...")
        sys.exit(0)

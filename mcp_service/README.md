# MCP Service

Model Context Protocol (MCP) Service - Standalone service for AI-powered trading operations.

## Structure

```
mcp_service/
├── main.py              # Entry point for MCP service
├── __init__.py          # Package initialization
├── server/              # MCP server implementation
│   ├── mcp_trading_server.py
│   └── __init__.py
├── llm/                 # LLM integration (Llama/Ollama)
│   ├── llama_integration.py
│   └── __init__.py
├── chat/                # Chat handling
│   ├── chat_handler.py
│   └── __init__.py
├── agents/              # Trading agents
│   ├── trading_agent.py
│   ├── explanation_agent.py
│   ├── insight_agent.py
│   ├── risk_agent.py
│   ├── portfolio_agent.py
│   └── __init__.py
├── tools/               # Trading tools
│   ├── execution_tool.py
│   ├── market_analysis_tool.py
│   ├── portfolio_tool.py
│   ├── risk_management_tool.py
│   ├── sentiment_tool.py
│   ├── prediction_tool.py
│   ├── scan_tool.py
│   └── __init__.py
└── config/              # Configuration
    └── __init__.py
```

## Running the Service

### Standalone Mode

```bash
python mcp_service/main.py
```

### Environment Variables

- `MCP_MONITORING_PORT`: Monitoring port (default: 8002)
- `MCP_MAX_SESSIONS`: Maximum concurrent sessions (default: 100)
- `LLAMA_BASE_URL`: Ollama base URL (default: http://localhost:11434)
- `LLAMA_MODEL`: Model name (default: llama3.1:8b)
- `LLAMA_MAX_TOKENS`: Maximum tokens (default: 2048)
- `LLAMA_TEMPERATURE`: Temperature for generation (default: 0.7)
- `CHAT_MAX_HISTORY`: Maximum chat history (default: 50)
- `ENABLE_ML`: Enable ML features (default: true)

## Features

- **LLM Integration**: Full Llama/Ollama integration for AI reasoning
- **Chat Handling**: Intelligent chat interface with command support
- **Trading Agents**: Advanced AI agents for trading decisions
- **Market Tools**: Comprehensive market analysis and execution tools
- **Standalone Operation**: Can run independently from web backend

## Integration

The MCP service can be integrated with the web backend or run as a standalone service. The web backend imports from `mcp_service` when available.


#!/usr/bin/env python3
"""
MCP Service API Server
======================

HTTP API server for MCP service to handle chat and LLM requests.
This allows web_backend to forward chat requests to the MCP service.
"""

import asyncio
import logging
import os
from typing import Dict, Any, Optional
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

logger = logging.getLogger(__name__)

# Global references to MCP components
chat_handler = None
trading_agent = None

# Request/Response models
class ChatRequest(BaseModel):
    message: str
    user_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    timestamp: str
    metadata: Optional[Dict[str, Any]] = None

# Create FastAPI app
app = FastAPI(title="MCP Service API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/chat", response_model=ChatResponse)
async def handle_chat(request: ChatRequest):
    """Handle chat requests - all chat processing happens here"""
    global chat_handler
    
    try:
        if not chat_handler:
            raise HTTPException(
                status_code=503,
                detail="Chat handler not initialized. MCP service may not be fully started."
            )
        
        logger.info(f"[MCP API] Processing chat message: {request.message[:50]}...")
        
        # Process message through chat handler
        response = await chat_handler.process_message(
            request.message,
            user_id=request.user_id
        )
        
        logger.info(f"[MCP API] Chat response generated successfully")
        
        return ChatResponse(
            response=response.response,
            timestamp=response.timestamp,
            metadata=response.metadata
        )
        
    except Exception as e:
        logger.error(f"[MCP API] Error processing chat: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Error processing chat request: {str(e)}"
        )

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if chat_handler else "not_ready",
        "chat_handler": chat_handler is not None,
        "trading_agent": trading_agent is not None,
        "timestamp": datetime.now().isoformat()
    }

def initialize_api(chat_hdlr, trading_agt=None):
    """Initialize API with MCP components"""
    global chat_handler, trading_agent
    chat_handler = chat_hdlr
    trading_agent = trading_agt
    logger.info("[MCP API] API server initialized with MCP components")

async def start_api_server(host: str = "0.0.0.0", port: int = 8003):
    """Start the API server"""
    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="info"
    )
    server = uvicorn.Server(config)
    logger.info(f"[MCP API] Starting API server on {host}:{port}")
    await server.serve()

if __name__ == "__main__":
    # For standalone testing
    asyncio.run(start_api_server())




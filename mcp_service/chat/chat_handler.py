#!/usr/bin/env python3
"""
Chat Handler for MCP Service
============================

Handles chat interactions, message processing, and responses.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import json

logger = logging.getLogger(__name__)

@dataclass
class ChatMessage:
    """Chat message structure"""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: str
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class ChatResponse:
    """Chat response structure"""
    response: str
    timestamp: str
    metadata: Optional[Dict[str, Any]] = None

class ChatHandler:
    """
    Chat Handler for processing user messages and generating responses
    
    Integrates with LLM and trading agents to provide intelligent responses.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Chat Handler
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.message_history: List[ChatMessage] = []
        self.max_history = config.get("max_history", 50)
        
        # Lazy imports to avoid circular dependencies
        self._groq_engine = None
        self._trading_agent = None
        
        logger.info("ChatHandler initialized")
    
    def _get_groq_engine(self):
        """Lazy import of Groq engine"""
        if self._groq_engine is None:
            try:
                from ..llm import GroqReasoningEngine
                groq_config = self.config.get("groq", {})
                self._groq_engine = GroqReasoningEngine(groq_config)
            except ImportError as e:
                logger.error(f"Failed to import GroqReasoningEngine: {e}")
                self._groq_engine = False
        return self._groq_engine if self._groq_engine is not False else None
    
    def _get_trading_agent(self):
        """Lazy import of trading agent"""
        if self._trading_agent is None:
            try:
                from ..agents import TradingAgent
                agent_config = self.config.get("trading_agent", {})
                self._trading_agent = TradingAgent(agent_config)
            except ImportError as e:
                logger.error(f"Failed to import TradingAgent: {e}")
                self._trading_agent = False
        return self._trading_agent if self._trading_agent is not False else None
    
    async def process_message(self, message: str, user_id: Optional[str] = None) -> ChatResponse:
        """
        Process a user message and generate response
        
        Args:
            message: User message
            user_id: Optional user identifier
            
        Returns:
            ChatResponse with generated response
        """
        try:
            # Add user message to history
            user_msg = ChatMessage(
                role="user",
                content=message,
                timestamp=datetime.now().isoformat()
            )
            self.message_history.append(user_msg)
            
            # Trim history if needed
            if len(self.message_history) > self.max_history:
                self.message_history = self.message_history[-self.max_history:]
            
            # Check for commands
            if message.startswith("/"):
                response = await self._handle_command(message)
            else:
                # Process as general chat
                response = await self._handle_general_chat(message)
            
            # Add assistant response to history
            assistant_msg = ChatMessage(
                role="assistant",
                content=response.response,
                timestamp=response.timestamp
            )
            self.message_history.append(assistant_msg)
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return ChatResponse(
                response=f"I encountered an error: {str(e)}",
                timestamp=datetime.now().isoformat()
            )
    
    async def _handle_command(self, command: str) -> ChatResponse:
        """Handle command messages"""
        try:
            parts = command.split()
            cmd = parts[0] if len(parts) > 0 else ""
            
            # Command routing
            if cmd == "/start_bot":
                return ChatResponse(
                    response="Bot is running and ready for trading!",
                    timestamp=datetime.now().isoformat()
                )
            elif cmd == "/get_pnl":
                # Get P&L information
                return ChatResponse(
                    response="Portfolio P&L information would be retrieved here.",
                    timestamp=datetime.now().isoformat()
                )
            elif cmd.startswith("/set_risk"):
                risk_level = parts[1] if len(parts) > 1 else "MEDIUM"
                return ChatResponse(
                    response=f"Risk level set to {risk_level}",
                    timestamp=datetime.now().isoformat()
                )
            else:
                return ChatResponse(
                    response=f"Unknown command: {cmd}. Type /help for available commands.",
                    timestamp=datetime.now().isoformat()
                )
        except Exception as e:
            logger.error(f"Error handling command: {e}")
            return ChatResponse(
                response=f"Error processing command: {str(e)}",
                timestamp=datetime.now().isoformat()
            )
    
    async def _handle_general_chat(self, message: str) -> ChatResponse:
        """Handle general chat messages"""
        try:
            groq_engine = self._get_groq_engine()
            
            if groq_engine:
                # Build context from history
                context = self._build_context()
                
                # Get response from LLM
                async with groq_engine:
                    result = await groq_engine.process_query(message, context)
                    response_text = result.get("response", "I couldn't generate a response.")
            else:
                response_text = "LLM engine not available. Please check configuration."
            
            return ChatResponse(
                response=response_text,
                timestamp=datetime.now().isoformat()
            )
        except Exception as e:
            logger.error(f"Error handling general chat: {e}")
            return ChatResponse(
                response=f"Error processing message: {str(e)}",
                timestamp=datetime.now().isoformat()
            )
    
    def _build_context(self) -> str:
        """Build context from message history"""
        if not self.message_history:
            return ""
        
        context_parts = []
        for msg in self.message_history[-10:]:  # Last 10 messages
            context_parts.append(f"{msg.role}: {msg.content}")
        
        return "\n".join(context_parts)
    
    def get_history(self, limit: Optional[int] = None) -> List[ChatMessage]:
        """Get message history"""
        if limit:
            return self.message_history[-limit:]
        return self.message_history
    
    def clear_history(self):
        """Clear message history"""
        self.message_history = []
        logger.info("Chat history cleared")


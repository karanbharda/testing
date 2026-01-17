#!/usr/bin/env python3
"""
Llama Integration for Trading Bot
=================================

Provides integration with Ollama/Llama models for AI-powered trading decisions.
"""

import asyncio
import logging
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import aiohttp

logger = logging.getLogger(__name__)

@dataclass
class TradingContext:
    """Context for trading decisions"""
    symbol: str
    current_price: float
    technical_signals: Dict[str, Any]
    market_data: Dict[str, Any]
    portfolio_data: Optional[Dict[str, Any]] = None
    risk_metrics: Optional[Dict[str, Any]] = None

@dataclass
class LlamaResponse:
    """Response from Llama model"""
    content: str
    reasoning: Optional[str] = None
    confidence: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

class LlamaReasoningEngine:
    """
    Llama Reasoning Engine for Trading Decisions
    
    Provides AI-powered reasoning using Ollama/Llama models.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Llama Reasoning Engine
        
        Args:
            config: Configuration dictionary with:
                - llama_base_url: Base URL for Ollama (default: http://localhost:11434)
                - llama_model: Model name (default: llama3.1:8b)
                - max_tokens: Maximum tokens (default: 2048)
                - temperature: Temperature for generation (default: 0.7)
        """
        self.base_url = config.get("llama_base_url", "http://localhost:11434")
        self.model = config.get("llama_model", "llama3.1:8b")
        self.max_tokens = config.get("max_tokens", 2048)
        self.temperature = config.get("temperature", 0.7)
        self.session: Optional[aiohttp.ClientSession] = None
        
        logger.info(f"LlamaReasoningEngine initialized with model: {self.model}")
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def _call_ollama(self, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Call Ollama API"""
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        url = f"{self.base_url}/api/generate"
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens
            }
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        try:
            async with self.session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=60)) as response:
                if response.status == 200:
                    result = await response.json()
                    return result
                else:
                    error_text = await response.text()
                    logger.error(f"Ollama API error: {response.status} - {error_text}")
                    raise Exception(f"Ollama API error: {response.status}")
        except asyncio.TimeoutError:
            logger.error("Ollama API timeout")
            raise Exception("Ollama API timeout")
        except Exception as e:
            logger.error(f"Error calling Ollama: {e}")
            raise
    
    async def process_query(self, message: str, enhanced_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a query and return response
        
        Args:
            message: User message
            enhanced_prompt: Optional enhanced prompt with context
            
        Returns:
            Dictionary with response and metadata
        """
        try:
            prompt = enhanced_prompt if enhanced_prompt else message
            
            system_prompt = """You are an expert trading assistant with deep knowledge of financial markets,
            technical analysis, and risk management. Provide clear, actionable insights based on the given context."""
            
            result = await self._call_ollama(prompt, system_prompt)
            
            response_text = result.get("response", "")
            
            return {
                "response": response_text,
                "model": self.model,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "response": f"I encountered an error: {str(e)}",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def explain_trade_decision(
        self,
        action: str,
        context: TradingContext
    ) -> LlamaResponse:
        """
        Explain a trading decision
        
        Args:
            action: Trading action (BUY/SELL/HOLD)
            context: Trading context
            
        Returns:
            LlamaResponse with explanation
        """
        try:
            prompt = f"""Explain the trading decision for {context.symbol}:
            
Action: {action}
Current Price: {context.current_price}
Technical Signals: {json.dumps(context.technical_signals, indent=2)}
Market Data: {json.dumps(context.market_data, indent=2)}

Provide a clear explanation of why this decision was made, considering:
1. Technical analysis signals
2. Market conditions
3. Risk factors
4. Portfolio considerations"""
            
            result = await self._call_ollama(prompt)
            response_text = result.get("response", "")
            
            return LlamaResponse(
                content=response_text,
                reasoning=response_text,
                confidence=0.8
            )
        except Exception as e:
            logger.error(f"Error explaining trade decision: {e}")
            return LlamaResponse(
                content=f"Error generating explanation: {str(e)}",
                reasoning=None
            )
    
    async def analyze_market_decision(
        self,
        symbol: str,
        context: TradingContext
    ) -> LlamaResponse:
        """
        Analyze market decision for a symbol
        
        Args:
            symbol: Stock symbol
            context: Trading context
            
        Returns:
            LlamaResponse with analysis
        """
        try:
            prompt = f"""Analyze the market decision for {symbol}:
            
Current Price: {context.current_price}
Technical Signals: {json.dumps(context.technical_signals, indent=2)}
Market Data: {json.dumps(context.market_data, indent=2)}

Provide a comprehensive analysis including:
1. Current market conditions
2. Technical indicators interpretation
3. Risk assessment
4. Recommended action"""
            
            result = await self._call_ollama(prompt)
            response_text = result.get("response", "")
            
            return LlamaResponse(
                content=response_text,
                reasoning=response_text,
                confidence=0.8
            )
        except Exception as e:
            logger.error(f"Error analyzing market decision: {e}")
            return LlamaResponse(
                content=f"Error in analysis: {str(e)}",
                reasoning=None
            )
    
    async def optimize_portfolio(self, portfolio_data: Dict[str, Any]) -> LlamaResponse:
        """
        Optimize portfolio allocation
        
        Args:
            portfolio_data: Portfolio information
            
        Returns:
            LlamaResponse with optimization suggestions
        """
        try:
            prompt = f"""Analyze and optimize the portfolio:
            
Portfolio Data: {json.dumps(portfolio_data, indent=2)}

Provide optimization recommendations including:
1. Asset allocation suggestions
2. Risk diversification
3. Rebalancing recommendations
4. Performance improvements"""
            
            result = await self._call_ollama(prompt)
            response_text = result.get("response", "")
            
            return LlamaResponse(
                content=response_text,
                reasoning=response_text
            )
        except Exception as e:
            logger.error(f"Error optimizing portfolio: {e}")
            return LlamaResponse(
                content=f"Error in portfolio optimization: {str(e)}",
                reasoning=None
            )
    
    async def generate_response(self, prompt: str) -> LlamaResponse:
        """
        Generate a general response
        
        Args:
            prompt: Input prompt
            
        Returns:
            LlamaResponse
        """
        try:
            result = await self._call_ollama(prompt)
            response_text = result.get("response", "")
            
            return LlamaResponse(
                content=response_text,
                reasoning=response_text
            )
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return LlamaResponse(
                content=f"Error generating response: {str(e)}",
                reasoning=None
            )
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of Ollama connection"""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            url = f"{self.base_url}/api/tags"
            async with self.session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                if response.status == 200:
                    return {
                        "status": "healthy",
                        "base_url": self.base_url,
                        "model": self.model
                    }
                else:
                    return {
                        "status": "unhealthy",
                        "error": f"HTTP {response.status}"
                    }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }


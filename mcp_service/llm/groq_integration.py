#!/usr/bin/env python3
"""
Groq Integration for Trading Bot
=================================

Provides integration with Groq models for AI-powered trading decisions.
"""

import asyncio
import logging
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import os
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
class GroqResponse:
    """Response from Groq model"""
    content: str
    reasoning: Optional[str] = None
    confidence: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

class GroqReasoningEngine:
    """
    Groq Reasoning Engine for Trading Decisions

    Provides AI-powered reasoning using Groq models.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Groq Reasoning Engine

        Args:
            config: Configuration dictionary with:
                - groq_api_key: Groq API key (from environment)
                - groq_model: Model name (default: llama3-8b-8192)
                - max_tokens: Maximum tokens (default: 2048)
                - temperature: Temperature for generation (default: 0.0 for deterministic output)
        """
        self.api_key = config.get("groq_api_key") or os.getenv("GROQ_API_KEY")
        self.available = bool(self.api_key)

        if self.available:
            self.model = config.get("groq_model", "llama-3.1-8b-instant")
            self.max_tokens = config.get("max_tokens", 2048)
            self.temperature = config.get("temperature", 0.0)  # Default to 0 for deterministic output
            self.session: Optional[aiohttp.ClientSession] = None
            logger.info(f"GroqReasoningEngine initialized with model: {self.model}")
        else:
            logger.warning("GROQ_API_KEY not found. GroqReasoningEngine will operate in stub mode.")
            self.model = "stub"
            self.max_tokens = 2048
            self.temperature = 0.0
            self.session = None

    def is_available(self) -> bool:
        """Check if the Groq engine is available (API key configured)"""
        return self.available

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

    async def _call_groq(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Call Groq API"""
        if not self.session:
            self.session = aiohttp.ClientSession()

        url = "https://api.groq.com/openai/v1/chat/completions"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }

        try:
            async with self.session.post(url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=60)) as response:
                if response.status == 200:
                    result = await response.json()
                    return result
                else:
                    error_text = await response.text()
                    logger.error(f"Groq API error: {response.status} - {error_text}")
                    raise Exception(f"Groq API error: {response.status}")
        except asyncio.TimeoutError:
            logger.error("Groq API timeout")
            raise Exception("Groq API timeout")
        except Exception as e:
            logger.error(f"Error calling Groq: {e}")
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
        # Handle stub mode when API key is not available
        if not self.available:
            return {
                "response": "LLM reasoning is not available. Please configure GROQ_API_KEY environment variable for enhanced AI reasoning capabilities.",
                "model": "stub",
                "available": False,
                "timestamp": datetime.now().isoformat()
            }

        try:
            prompt = enhanced_prompt if enhanced_prompt else message

            system_prompt = """You are a financial explanation narrator for institutional-grade trading systems.

CRITICAL RULES - VIOLATION PROHIBITED:
- You must not add new reasons, logic, or information
- You must not introduce predictions, assumptions, or speculation
- You must not use AI-related language (no "AI", "model", "algorithm", "intelligence")
- You must not use confidence language ("confident", "likely", "probably", "expected")
- You must not use promotional language ("excellent", "great", "amazing", "opportunity")
- You must only rewrite the given explanation into clear, professional financial language
- Keep explanation factual, neutral, and compliant with SEBI regulations
- Use only terms from the provided explanation blocks
- Maintain the exact meaning and intent of the original explanation

SEBI COMPLIANCE REQUIREMENTS:
- No forward-looking statements
- No investment advice or recommendations
- No market predictions or price targets
- No performance guarantees or projections
- Use neutral, factual language only
- Focus on risk management and capital protection

RESPONSE FORMAT:
- Professional financial terminology
- Clear and concise explanations
- Factual statements only
- No emotional or speculative language"""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]

            result = await self._call_groq(messages)

            response_text = result["choices"][0]["message"]["content"]

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
    ) -> GroqResponse:
        """
        Explain a trading decision

        Args:
            action: Trading action (BUY/SELL/HOLD)
            context: Trading context

        Returns:
            GroqResponse with explanation
        """
        # Handle stub mode when API key is not available
        if not self.available:
            return GroqResponse(
                content=f"Trade decision explanation not available. Configure GROQ_API_KEY for AI-powered explanations. Action: {action}",
                reasoning="LLM not configured",
                confidence=0.0
            )

        try:
            system_prompt = """You are a financial explanation narrator for institutional-grade trading systems.

CRITICAL RULES - VIOLATION PROHIBITED:
- You must not add new reasons, logic, or information
- You must not introduce predictions, assumptions, or speculation
- You must not use AI-related language (no "AI", "model", "algorithm", "intelligence")
- You must not use confidence language ("confident", "likely", "probably", "expected")
- You must not use promotional language ("excellent", "great", "amazing", "opportunity")
- You must only rewrite the given explanation into clear, professional financial language
- Keep explanation factual, neutral, and compliant with SEBI regulations
- Use only terms from the provided explanation blocks
- Maintain the exact meaning and intent of the original explanation

SEBI COMPLIANCE REQUIREMENTS:
- No forward-looking statements
- No investment advice or recommendations
- No market predictions or price targets
- No performance guarantees or projections
- Use neutral, factual language only
- Focus on risk management and capital protection

RESPONSE FORMAT:
- Professional financial terminology
- Clear and concise explanations
- Factual statements only
- No emotional or speculative language"""

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

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]

            result = await self._call_groq(messages)
            response_text = result["choices"][0]["message"]["content"]

            return GroqResponse(
                content=response_text,
                reasoning=response_text,
                confidence=0.8
            )
        except Exception as e:
            logger.error(f"Error explaining trade decision: {e}")
            return GroqResponse(
                content=f"Error generating explanation: {str(e)}",
                reasoning=None
            )

    async def analyze_market_decision(
        self,
        symbol: str,
        context: TradingContext
    ) -> GroqResponse:
        """
        Analyze market decision for a symbol

        Args:
            symbol: Stock symbol
            context: Trading context

        Returns:
            GroqResponse with analysis
        """
        # Handle stub mode when API key is not available
        if not self.available:
            return GroqResponse(
                content=f"Market analysis not available. Configure GROQ_API_KEY for AI-powered market analysis. Symbol: {symbol}",
                reasoning="LLM not configured",
                confidence=0.0
            )

        try:
            system_prompt = """You are a financial explanation narrator.

Rules:
- You must not add new reasons.
- You must not introduce new logic.
- You must not use AI-related language.
- You must only rewrite the given explanation into clear professional financial language.
- Keep explanation factual and neutral.

Do not include opinions.
Do not include predictions.
Do not include assumptions."""

            prompt = f"""Analyze the market decision for {symbol}:

Current Price: {context.current_price}
Technical Signals: {json.dumps(context.technical_signals, indent=2)}
Market Data: {json.dumps(context.market_data, indent=2)}

Provide a comprehensive analysis including:
1. Current market conditions
2. Technical indicators interpretation
3. Risk assessment
4. Recommended action"""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]

            result = await self._call_groq(messages)
            response_text = result["choices"][0]["message"]["content"]

            return GroqResponse(
                content=response_text,
                reasoning=response_text,
                confidence=0.8
            )
        except Exception as e:
            logger.error(f"Error analyzing market decision: {e}")
            return GroqResponse(
                content=f"Error in analysis: {str(e)}",
                reasoning=None
            )

    async def optimize_portfolio(self, portfolio_data: Dict[str, Any]) -> GroqResponse:
        """
        Optimize portfolio allocation

        Args:
            portfolio_data: Portfolio information

        Returns:
            GroqResponse with optimization suggestions
        """
        # Handle stub mode when API key is not available
        if not self.available:
            return GroqResponse(
                content="Portfolio optimization not available. Configure GROQ_API_KEY for AI-powered portfolio optimization.",
                reasoning="LLM not configured",
                confidence=0.0
            )

        try:
            system_prompt = """You are a financial explanation narrator for institutional-grade trading systems.

CRITICAL RULES - VIOLATION PROHIBITED:
- You must not add new reasons, logic, or information
- You must not introduce predictions, assumptions, or speculation
- You must not use AI-related language (no "AI", "model", "algorithm", "intelligence")
- You must not use confidence language ("confident", "likely", "probably", "expected")
- You must not use promotional language ("excellent", "great", "amazing", "opportunity")
- You must only rewrite the given explanation into clear, professional financial language
- Keep explanation factual, neutral, and compliant with SEBI regulations
- Use only terms from the provided explanation blocks
- Maintain the exact meaning and intent of the original explanation

SEBI COMPLIANCE REQUIREMENTS:
- No forward-looking statements
- No investment advice or recommendations
- No market predictions or price targets
- No performance guarantees or projections
- Use neutral, factual language only
- Focus on risk management and capital protection

RESPONSE FORMAT:
- Professional financial terminology
- Clear and concise explanations
- Factual statements only
- No emotional or speculative language"""

            prompt = f"""Analyze and optimize the portfolio:

Portfolio Data: {json.dumps(portfolio_data, indent=2)}

Provide optimization recommendations including:
1. Asset allocation suggestions
2. Risk diversification
3. Rebalancing recommendations
4. Performance improvements"""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]

            result = await self._call_groq(messages)
            response_text = result["choices"][0]["message"]["content"]

            return GroqResponse(
                content=response_text,
                reasoning=response_text
            )
        except Exception as e:
            logger.error(f"Error optimizing portfolio: {str(e)}")
            return GroqResponse(
                content=f"Error in portfolio optimization: {str(e)}",
                reasoning=None
            )

    async def generate_response(self, prompt: str) -> GroqResponse:
        """
        Generate a general response

        Args:
            prompt: Input prompt

        Returns:
            GroqResponse
        """
        # Handle stub mode when API key is not available
        if not self.available:
            return GroqResponse(
                content="Response generation not available. Configure GROQ_API_KEY for AI-powered responses.",
                reasoning="LLM not configured",
                confidence=0.0
            )

        try:
            system_prompt = """You are a financial explanation narrator for institutional-grade trading systems.

CRITICAL RULES - VIOLATION PROHIBITED:
- You must not add new reasons, logic, or information
- You must not introduce predictions, assumptions, or speculation
- You must not use AI-related language (no "AI", "model", "algorithm", "intelligence")
- You must not use confidence language ("confident", "likely", "probably", "expected")
- You must not use promotional language ("excellent", "great", "amazing", "opportunity")
- You must only rewrite the given explanation into clear, professional financial language
- Keep explanation factual, neutral, and compliant with SEBI regulations
- Use only terms from the provided explanation blocks
- Maintain the exact meaning and intent of the original explanation

SEBI COMPLIANCE REQUIREMENTS:
- No forward-looking statements
- No investment advice or recommendations
- No market predictions or price targets
- No performance guarantees or projections
- Use neutral, factual language only
- Focus on risk management and capital protection

RESPONSE FORMAT:
- Professional financial terminology
- Clear and concise explanations
- Factual statements only
- No emotional or speculative language"""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]

            result = await self._call_groq(messages)
            response_text = result["choices"][0]["message"]["content"]

            return GroqResponse(
                content=response_text,
                reasoning=response_text
            )
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return GroqResponse(
                content=f"Error generating response: {str(e)}",
                reasoning=None
            )

    async def health_check(self) -> Dict[str, Any]:
        """Check health of Groq connection"""
        if not self.available:
            return {
                "status": "stub_mode",
                "api_key_configured": False,
                "message": "GROQ_API_KEY not configured. Operating in stub mode."
            }

        try:
            if not self.session:
                self.session = aiohttp.ClientSession()

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            # Simple test call to check API connectivity
            test_messages = [
                {"role": "user", "content": "Hello"}
            ]

            payload = {
                "model": self.model,
                "messages": test_messages,
                "max_tokens": 10
            }

            url = "https://api.groq.com/openai/v1/chat/completions"
            async with self.session.post(url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    return {
                        "status": "healthy",
                        "api_key_configured": bool(self.api_key),
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
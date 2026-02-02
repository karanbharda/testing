#!/usr/bin/env python3
"""
Finance Chat Grounding System
=============================

Grounds chatbot responses in financial logic, market reasoning, and risk language.
Integrates RAG for knowledge retrieval and trade explanation.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import re
import sys
from pathlib import Path

# Add financeKnowlegde to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "financeKnowlegde"))

from financeKnowlegde.vectorstore.rag_loader import FinanceRAGLoader

logger = logging.getLogger(__name__)

class FinanceChatGrounding:
    """
    Finance Chat Grounding System

    Provides grounded responses for:
    - Finance definitions
    - Trade explanations
    - Sentiment context
    - Risk gate compliance

    Explainability Rules:
    - ONLY financial logic, market reasoning, risk language
    - NEVER mention model confidence, thresholds, internal scores, agent logic, model weights
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Finance Chat Grounding

        Args:
            config: Configuration with rag_config, etc.
        """
        self.config = config

        # Initialize RAG system
        rag_config = config.get("rag", {})
        self.rag_loader = FinanceRAGLoader(rag_config)

        # Knowledge base path
        self.kb_path = Path(config.get("kb_path", "Finance_KB"))

        # Grounding rules
        self.allowed_patterns = [
            r'price.*resistance|support',
            r'volume.*declining|increasing',
            r'risk.*rules.*exit|entry',
            r'sector.*sentiment.*weak|strong',
            r'market.*conditions.*favorable|adverse',
            r'technical.*signals.*bullish|bearish',
            r'position.*sizing.*risk',
            r'stop.*loss.*triggered|activated',
            r'volatility.*spike|surge',
            r'margin.*requirements.*breached',
            r'circuit.*breaker.*activated',
            r'drawdown.*limit.*hit|exceeded'
        ]

        self.forbidden_patterns = [
            r'confidence.*score|level',
            r'threshold.*below|above',
            r'internal.*score|metric',
            r'agent.*logic|decision',
            r'model.*weight|parameter',
            r'probability.*\d+\.?\d*%',
            r'algorithm.*determined|calculated',
            r'AI.*analysis|assessment',
            r'machine.*learning.*prediction',
            r'neural.*network.*output'
        ]

    async def ground_response(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate grounded response for finance query

        Args:
            query: User query
            context: Additional context (trade data, market conditions, etc.)

        Returns:
            Grounded response with explanation
        """
        try:
            # Retrieve relevant knowledge
            relevant_chunks = self.rag_loader.retrieve(query, top_k=3)

            # Build context from retrieved knowledge
            knowledge_context = self._build_knowledge_context(relevant_chunks, query)

            # If RAG didn't provide good knowledge, try direct file search
            if not knowledge_context or len(knowledge_context.strip()) < 20:
                direct_knowledge = self._get_direct_knowledge(query)
                if direct_knowledge:
                    knowledge_context = direct_knowledge

            # Generate grounded response
            response = self._generate_grounded_response(query, knowledge_context, context)

            # Validate response against explainability rules
            validation = self._validate_explainability(response)

            return {
                'response': response,
                'validation': validation,
                'knowledge_sources': len(relevant_chunks),
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error in grounded response: {e}")
            return {
                'response': "Unable to provide explanation at this time.",
                'validation': {'compliant': False, 'issues': [str(e)]},
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def _build_knowledge_context(self, chunks: List[Dict[str, Any]], query: str) -> str:
        """Build knowledge context from retrieved chunks, filtered by query relevance"""
        if not chunks:
            return ""

        # Score chunks by query relevance
        scored_chunks = []
        query_lower = query.lower()

        for chunk in chunks:
            score = chunk['score']
            text_lower = chunk['text'].lower()
            source_lower = chunk['source'].lower()

            # Boost score for relevant keywords
            relevance_boost = 0

            if 'nse' in query_lower and ('nse' in text_lower or 'nse' in source_lower):
                relevance_boost += 0.5
            if 'circuit' in query_lower and 'circuit' in text_lower:
                relevance_boost += 0.5
            if 'settlement' in query_lower and 'settlement' in text_lower:
                relevance_boost += 0.5
            if 'trading hours' in query_lower and ('trading' in text_lower or 'hours' in text_lower):
                relevance_boost += 0.5
            if 'margin' in query_lower and 'margin' in text_lower:
                relevance_boost += 0.5
            if 'span' in query_lower and 'span' in text_lower:
                relevance_boost += 0.5
            if 'mtm' in query_lower and 'mtm' in text_lower:
                relevance_boost += 0.5

            if 'f&o' in query_lower or 'ban' in query_lower or 'derivative' in query_lower:
                if 'f&o' in text_lower or 'fno' in source_lower or 'ban' in text_lower:
                    relevance_boost += 0.5

            if 'stop loss' in query_lower or 'stop' in query_lower:
                if 'stop' in text_lower or 'stop_loss' in source_lower:
                    relevance_boost += 0.5

            if 'regime' in query_lower or 'market' in query_lower:
                if 'regime' in text_lower or 'market' in text_lower:
                    relevance_boost += 0.3

            if 'risk' in query_lower or 'exposure' in query_lower or 'liquidat' in query_lower:
                if 'risk' in text_lower or 'exposure' in text_lower:
                    relevance_boost += 0.3

            final_score = score + relevance_boost
            scored_chunks.append((final_score, chunk))

        # Sort by final score and take top chunks
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        top_chunks = [chunk for score, chunk in scored_chunks[:2]]  # Take top 2 most relevant

        # Extract clean text content (remove markdown headers)
        context_parts = []
        for chunk in top_chunks:
            text = chunk['text']
            # Remove markdown headers
            lines = text.split('\n')
            clean_lines = []
            for line in lines:
                line = line.strip()
                if line.startswith('#'):  # Skip headers
                    continue
                if line and not line.startswith('##'):  # Skip subheaders
                    clean_lines.append(line)

            clean_text = ' '.join(clean_lines)
            if clean_text.strip():
                context_parts.append(clean_text)

        return ' '.join(context_parts)

    def _get_direct_knowledge(self, query: str) -> str:
        """Get knowledge directly from files using keyword matching"""
        query_lower = query.lower()

        # Define keyword mappings to files and sections
        knowledge_map = {
            'fno_basics.md': {
                'keywords': ['f&o', 'futures', 'options', 'derivative', 'ban', 'trading restrictions'],
                'sections': {
                    'ban': 'F&O Ban Mechanism',
                    'margin': 'Margin Requirements',
                    'position': 'Position Limits'
                }
            },
            'stop_loss_rules.md': {
                'keywords': ['stop loss', 'stop', 'trailing', 'volatility', 'atr'],
                'sections': {
                    'trigger': 'Stop Loss Mechanisms',
                    'types': 'Types of Stop Loss',
                    'sebi': 'SEBI Requirements for Stop Loss'
                }
            },
            'nse_basics.md': {
                'keywords': ['nse', 'circuit', 'settlement', 'trading hours', 'margin', 'span', 'mtm'],
                'sections': {
                    'circuit': 'Circuit Breaker System',
                    'settlement': 'Settlement Process',
                    'trading': 'Trading Hours',
                    'margin': 'Margin Requirements'
                }
            }
        }

        # Find relevant file
        relevant_file = None
        for filename, config in knowledge_map.items():
            if any(keyword in query_lower for keyword in config['keywords']):
                relevant_file = filename
                break

        if not relevant_file:
            return ""

        # Read the file and extract relevant section
        file_path = self.kb_path / relevant_file.replace('_', '/').replace('.md', '.md')
        if not file_path.exists():
            # Try different path structure
            file_path = self.kb_path / relevant_file

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Extract relevant section based on query
            lines = content.split('\n')
            relevant_section = []
            in_section = False

            for line in lines:
                line_lower = line.lower()
                if line.startswith('##'):
                    # Check if this section matches query
                    section_title = line[2:].strip().lower()
                    if any(keyword in section_title for keyword in query_lower.split()) or \
                       any(keyword in query_lower for keyword in section_title.split()):
                        in_section = True
                        relevant_section = [line]  # Start new section
                    else:
                        in_section = False
                elif in_section and line.strip():
                    relevant_section.append(line)

            if relevant_section:
                return '\n'.join(relevant_section)

            # If no specific section, return general content
            return content[:500] + "..." if len(content) > 500 else content

        except Exception as e:
            logger.error(f"Error reading knowledge file {file_path}: {e}")
            return ""
        """Get knowledge directly from files using keyword matching"""
        query_lower = query.lower()

        # Define keyword mappings to files and sections
        knowledge_map = {
            'fno_basics.md': {
                'keywords': ['f&o', 'futures', 'options', 'derivative', 'ban', 'trading restrictions'],
                'sections': {
                    'ban': 'F&O Ban Mechanism',
                    'margin': 'Margin Requirements',
                    'position': 'Position Limits'
                }
            },
            'stop_loss_rules.md': {
                'keywords': ['stop loss', 'stop', 'trailing', 'volatility', 'atr'],
                'sections': {
                    'trigger': 'Stop Loss Mechanisms',
                    'types': 'Types of Stop Loss',
                    'sebi': 'SEBI Requirements for Stop Loss'
                }
            },
            'nse_basics.md': {
                'keywords': ['nse', 'circuit', 'settlement', 'trading hours', 'margin', 'span', 'mtm'],
                'sections': {
                    'circuit': 'Circuit Breaker System',
                    'settlement': 'Settlement Process',
                    'trading': 'Trading Hours',
                    'margin': 'Margin Requirements'
                }
            }
        }

        # Find relevant file
        relevant_file = None
        for filename, config in knowledge_map.items():
            if any(keyword in query_lower for keyword in config['keywords']):
                relevant_file = filename
                break

        if not relevant_file:
            return ""

        # Read the file and extract relevant section
        file_path = self.kb_path / relevant_file.replace('_', '/').replace('.md', '.md')
        if not file_path.exists():
            # Try different path structure
            file_path = self.kb_path / relevant_file

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Extract relevant section based on query
            lines = content.split('\n')
            relevant_section = []
            in_section = False

            for line in lines:
                line_lower = line.lower()
                if line.startswith('##'):
                    # Check if this section matches query
                    section_title = line[2:].strip().lower()
                    if any(keyword in section_title for keyword in query_lower.split()) or \
                       any(keyword in query_lower for keyword in section_title.split()):
                        in_section = True
                        relevant_section = [line]  # Start new section
                    else:
                        in_section = False
                elif in_section and line.strip():
                    relevant_section.append(line)

            if relevant_section:
                return '\n'.join(relevant_section)

            # If no specific section, return general content
            return content[:500] + "..." if len(content) > 500 else content

        except Exception as e:
            logger.error(f"Error reading knowledge file {file_path}: {e}")
            return ""

    def _generate_grounded_response(self, query: str, knowledge: str, context: Optional[Dict[str, Any]]) -> str:
        """Generate response grounded in financial logic"""

        # Classify query type
        query_type = self._classify_query(query)

        if query_type == 'definition':
            return self._explain_definition(query, knowledge)
        elif query_type == 'trade_explanation':
            return self._explain_trade(query, knowledge, context or {})
        elif query_type == 'risk_assessment':
            return self._assess_risk(query, knowledge, context or {})
        elif query_type == 'market_mechanics':
            return self._explain_mechanics(query, knowledge)
        else:
            return self._provide_general_guidance(query, knowledge)

    def _classify_query(self, query: str) -> str:
        """Classify the type of finance query"""
        query_lower = query.lower()

        if any(word in query_lower for word in ['what is', 'define', 'meaning of', 'explain']):
            return 'definition'
        elif any(word in query_lower for word in ['why did', 'trade', 'position', 'exit', 'entry']):
            return 'trade_explanation'
        elif any(word in query_lower for word in ['risk', 'stop loss', 'margin', 'exposure']):
            return 'risk_assessment'
        elif any(word in query_lower for word in ['how does', 'mechanics', 'process', 'work']):
            return 'market_mechanics'
        else:
            return 'general'

    def _explain_definition(self, query: str, knowledge: str) -> str:
        """Provide grounded definition explanation"""
        # Use knowledge from RAG if available
        if knowledge and len(knowledge.strip()) > 10:
            # Extract relevant parts from knowledge
            if 'circuit' in query.lower() and 'circuit' in knowledge.lower():
                return knowledge[:200] + "..." if len(knowledge) > 200 else knowledge
            elif 'settlement' in query.lower() and 'settlement' in knowledge.lower():
                return knowledge[:200] + "..." if len(knowledge) > 200 else knowledge
            elif 'margin' in query.lower() and 'margin' in knowledge.lower():
                return knowledge[:200] + "..." if len(knowledge) > 200 else knowledge
            elif 'trading hours' in query.lower() and ('trading' in knowledge.lower() or 'hours' in knowledge.lower()):
                return knowledge[:200] + "..." if len(knowledge) > 200 else knowledge
            else:
                # Return knowledge directly if relevant
                return knowledge[:300] + "..." if len(knowledge) > 300 else knowledge

        # Fallback to basic definitions
        if 'nse' in query.lower():
            return "NSE operates India's primary stock exchange with electronic trading systems. It handles equity, derivatives, and debt market transactions under SEBI regulation."
        elif 'stop loss' in query.lower():
            return "Stop loss orders automatically exit positions when price moves against the trade beyond acceptable risk levels, protecting capital from excessive losses."
        elif 'f&o' in query.lower() or 'futures' in query.lower():
            return "F&O markets allow trading in derivative contracts based on underlying assets. NSE IF&O provides futures and options on indices, stocks, and currencies."
        else:
            return "This financial concept involves market mechanics and risk management principles specific to Indian regulatory framework."

    def _explain_trade(self, query: str, knowledge: str, context: Dict[str, Any]) -> str:
        """Explain trade decisions in financial terms"""
        # Use knowledge from RAG if available and relevant to query
        if 'ban' in query.lower() and 'ban' in knowledge.lower():
            return knowledge[:250] + "..." if len(knowledge) > 250 else knowledge
        elif 'stop loss' in query.lower() and 'stop' in knowledge.lower():
            return knowledge[:250] + "..." if len(knowledge) > 250 else knowledge
        elif 'liquidat' in query.lower() and ('risk' in knowledge.lower() or 'exposure' in knowledge.lower()):
            return knowledge[:250] + "..." if len(knowledge) > 250 else knowledge

        # Build explanation from context
        explanations = []

        if context.get('stop_loss_hit'):
            explanations.append("Position exited when stop loss triggered due to adverse price movement")

        if context.get('volatility_spike'):
            explanations.append("Volatility surge activated risk management protocols")

        if context.get('volume_declining'):
            explanations.append("Declining volume indicated weakening market participation")

        if context.get('sector_sentiment') == 'weak':
            explanations.append("Sector sentiment deteriorated, prompting position adjustment")

        if not explanations:
            explanations.append("Market conditions changed, requiring position management")

        return ". ".join(explanations) + "."

    def _assess_risk(self, query: str, knowledge: str, context: Dict[str, Any]) -> str:
        """Provide risk assessment in financial terms"""
        risk_factors = []

        if context.get('position_size_large'):
            risk_factors.append("Position size exceeded recommended risk limits")

        if context.get('volatility_high'):
            risk_factors.append("Market volatility increased position risk profile")

        if context.get('margin_low'):
            risk_factors.append("Margin requirements not met for current exposure")

        if context.get('concentration_high'):
            risk_factors.append("Portfolio concentration created elevated risk")

        if not risk_factors:
            risk_factors.append("Standard risk management protocols applied")

        return "Risk assessment identified: " + "; ".join(risk_factors) + "."

    def _explain_mechanics(self, query: str, knowledge: str) -> str:
        """Explain market mechanics"""
        # Use knowledge from RAG if available
        if knowledge and len(knowledge.strip()) > 10:
            if 'settlement' in query.lower() and 'settlement' in knowledge.lower():
                return knowledge[:250] + "..." if len(knowledge) > 250 else knowledge
            elif 'margin' in query.lower() and 'margin' in knowledge.lower():
                return knowledge[:250] + "..." if len(knowledge) > 250 else knowledge
            elif 'circuit' in query.lower() and 'circuit' in knowledge.lower():
                return knowledge[:250] + "..." if len(knowledge) > 250 else knowledge
            elif 'trading' in query.lower() and 'trading' in knowledge.lower():
                return knowledge[:250] + "..." if len(knowledge) > 250 else knowledge
            else:
                return knowledge[:300] + "..." if len(knowledge) > 300 else knowledge

        # Fallback responses
        if 'settlement' in query.lower():
            return "NSE follows T+2 rolling settlement cycle. Pay-in occurs T+1, pay-out on T+2. Securities delivered in dematerialized form through depositories."
        elif 'margin' in query.lower():
            return "SPAN margin calculates worst-case loss scenarios. Exposure margins add buffer for concentrated positions. MTM settles daily profits and losses."
        elif 'circuit' in query.lower():
            return "Circuit breakers halt trading at ±10%, ±15%, ±20% levels. Level 1 allows 15-minute cooling period, Level 3 stops trading for the day."
        else:
            return "Market mechanics follow SEBI regulations ensuring fair and transparent trading with risk management controls."

    def _provide_general_guidance(self, query: str, knowledge: str) -> str:
        """Provide general financial guidance"""
        # Use knowledge from RAG if available
        if knowledge and len(knowledge.strip()) > 10:
            return knowledge[:300] + "..." if len(knowledge) > 300 else knowledge

        return "Market operations follow established financial principles and regulatory guidelines. Risk management remains the primary consideration in all market activities."

    def _validate_explainability(self, response: str) -> Dict[str, Any]:
        """Validate response against explainability rules"""
        issues = []

        # Check for forbidden patterns
        for pattern in self.forbidden_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                issues.append(f"Contains forbidden pattern: {pattern}")

        # Check for allowed financial language
        has_financial_language = any(
            re.search(pattern, response, re.IGNORECASE)
            for pattern in self.allowed_patterns
        )

        if not has_financial_language and len(response.split()) > 5:
            issues.append("Response lacks financial reasoning language")

        return {
            'compliant': len(issues) == 0,
            'issues': issues,
            'financial_language_detected': has_financial_language
        }

    async def explain_trade_decision(self, trade_data: Dict[str, Any]) -> str:
        """
        Explain a specific trade decision

        Args:
            trade_data: Trade execution data

        Returns:
            Grounded explanation
        """
        context = {
            'stop_loss_hit': trade_data.get('stop_loss_triggered', False),
            'volatility_spike': trade_data.get('volatility_spike', False),
            'volume_declining': trade_data.get('volume_trend') == 'declining',
            'sector_sentiment': trade_data.get('sector_sentiment', 'neutral'),
            'position_size_large': trade_data.get('position_size', 0) > 100000,
            'volatility_high': trade_data.get('volatility', 0) > 30,
            'margin_low': trade_data.get('margin_available', 100) < 50
        }

        result = await self.ground_response("Explain this trade decision", context)
        return result['response']
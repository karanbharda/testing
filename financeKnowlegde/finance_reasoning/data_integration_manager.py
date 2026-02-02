#!/usr/bin/env python3
"""
Finance KB Data Integration & Synchronization System
=====================================================

Manages data integration from external sources:
- Market data feeds
- Corporate actions
- Regulatory updates
- Historical data

Ensures KB stays current with industry standards.
"""

import json
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime, timedelta
import hashlib

logger = logging.getLogger(__name__)

class DataIntegrationManager:
    """Manages data integration for Finance KB"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.kb_path = Path(config.get("kb_path", "Finance_KB"))
        self.integration_log_path = Path(config.get("integration_log_path", "data/kb_integration.json"))
        
        # Metadata for tracking
        self.metadata_file = self.kb_path / "integration_metadata.json"
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> Dict[str, Any]:
        """Load integration metadata"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error loading metadata: {e}")
        
        return {
            'version': '2.0.0',
            'last_update': datetime.now().isoformat(),
            'sources': {},
            'categories': {}
        }

    def _save_metadata(self):
        """Save integration metadata"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")

    def add_market_data(self, symbol: str, data: Dict[str, Any], category: str = 'equities') -> bool:
        """
        Add market data for stock/instrument
        
        Args:
            symbol: Trading symbol
            data: Market data (price, volume, volatility, etc.)
            category: KB category
            
        Returns:
            Success status
        """
        try:
            # Create content from market data
            content = self._format_market_data(symbol, data)
            
            # Write to appropriate category file
            file_path = self.kb_path / category / f"{symbol.lower()}_market_data.md"
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'w') as f:
                f.write(content)
            
            # Update metadata
            self.metadata['sources'][symbol] = {
                'last_update': datetime.now().isoformat(),
                'category': category,
                'data_type': 'market_data'
            }
            self._save_metadata()
            
            logger.info(f"Added market data for {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding market data for {symbol}: {e}")
            return False

    def add_corporate_action(self, symbol: str, action: Dict[str, Any]) -> bool:
        """
        Add corporate action information
        
        Args:
            symbol: Trading symbol
            action: Corporate action details (dividend, split, bonus, etc.)
            
        Returns:
            Success status
        """
        try:
            content = f"""# Corporate Action - {symbol}

## Action Type
{action.get('type', 'Unknown')}

## Details
- Date: {action.get('date', 'N/A')}
- Ratio: {action.get('ratio', 'N/A')}
- Ex-Date: {action.get('ex_date', 'N/A')}
- Record Date: {action.get('record_date', 'N/A')}
- Payment Date: {action.get('payment_date', 'N/A')}

## Impact
{action.get('impact', 'No impact details provided')}

## Historical Context
Previous corporate actions and patterns for reference.
"""
            
            file_path = self.kb_path / 'equities' / f"{symbol.lower()}_corporate_actions.md"
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'a') as f:
                f.write("\n\n" + content)
            
            logger.info(f"Added corporate action for {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding corporate action: {e}")
            return False

    def add_regulatory_update(self, title: str, content: str, update_type: str = 'sebi') -> bool:
        """
        Add regulatory update to KB
        
        Args:
            title: Update title
            content: Update content
            update_type: Type of regulation (sebi, nse, rbi, etc.)
            
        Returns:
            Success status
        """
        try:
            # Determine appropriate category
            category_map = {
                'sebi': 'equities',
                'nse': 'equities',
                'rbi': 'macro',
                'monetary': 'macro',
                'policy': 'macro'
            }
            category = category_map.get(update_type.lower(), 'equities')
            
            # Create filename from title
            filename = f"{update_type}_{self._slugify(title)}.md"
            file_path = self.kb_path / category / filename
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Format content
            formatted_content = f"""# {title}

**Type:** {update_type}
**Date:** {datetime.now().isoformat()}

{content}

---
*Last updated: {datetime.now().isoformat()}*
"""
            
            with open(file_path, 'w') as f:
                f.write(formatted_content)
            
            logger.info(f"Added regulatory update: {title}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding regulatory update: {e}")
            return False

    def add_strategy_update(self, strategy_name: str, content: str, performance: Dict[str, Any] = None) -> bool:
        """
        Add or update strategy documentation
        
        Args:
            strategy_name: Name of strategy
            content: Strategy details and rules
            performance: Optional performance metrics
            
        Returns:
            Success status
        """
        try:
            file_path = self.kb_path / 'strategies' / f"{self._slugify(strategy_name)}.md"
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Add performance section if provided
            perf_section = ""
            if performance:
                perf_section = f"""
## Performance Metrics
- Win Rate: {performance.get('win_rate', 'N/A')}%
- Profit Factor: {performance.get('profit_factor', 'N/A')}x
- Sharpe Ratio: {performance.get('sharpe_ratio', 'N/A')}
- Max Drawdown: {performance.get('max_drawdown', 'N/A')}%
"""
            
            formatted_content = f"""# {strategy_name}

{content}
{perf_section}

---
*Last updated: {datetime.now().isoformat()}*
"""
            
            with open(file_path, 'w') as f:
                f.write(formatted_content)
            
            logger.info(f"Added/updated strategy: {strategy_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding strategy: {e}")
            return False

    def add_indicator_analysis(self, indicator_name: str, analysis: Dict[str, Any]) -> bool:
        """
        Add technical indicator analysis or update
        
        Args:
            indicator_name: Indicator name (RSI, MACD, etc.)
            analysis: Analysis details
            
        Returns:
            Success status
        """
        try:
            file_path = self.kb_path / 'ta_indicators' / f"{indicator_name.lower()}_analysis.md"
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            content = f"""# {indicator_name} Analysis & Application

## Overview
{analysis.get('overview', '')}

## Calculation
{analysis.get('calculation', '')}

## Trading Rules
{analysis.get('trading_rules', '')}

## Limitations
{analysis.get('limitations', '')}

## Performance in Different Market Conditions
{analysis.get('market_conditions', '')}

---
*Last updated: {datetime.now().isoformat()}*
"""
            
            with open(file_path, 'w') as f:
                f.write(content)
            
            logger.info(f"Added indicator analysis: {indicator_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding indicator analysis: {e}")
            return False

    def _format_market_data(self, symbol: str, data: Dict[str, Any]) -> str:
        """Format market data as markdown"""
        return f"""# Market Data - {symbol}

## Current Price Data
- Current Price: ₹{data.get('price', 'N/A')}
- 52-Week High: ₹{data.get('high_52w', 'N/A')}
- 52-Week Low: ₹{data.get('low_52w', 'N/A')}
- Market Cap: {data.get('market_cap', 'N/A')}

## Volatility Metrics
- Beta: {data.get('beta', 'N/A')}
- Volatility (Annualized): {data.get('volatility', 'N/A')}%
- Average Volume: {data.get('avg_volume', 'N/A')}
- Volume Trend: {data.get('volume_trend', 'N/A')}

## Technical Levels
- Resistance: ₹{data.get('resistance', 'N/A')}
- Support: ₹{data.get('support', 'N/A')}

## Fundamentals
- P/E Ratio: {data.get('pe_ratio', 'N/A')}
- Dividend Yield: {data.get('div_yield', 'N/A')}%

---
*Last updated: {datetime.now().isoformat()}*
"""

    def _slugify(self, text: str) -> str:
        """Convert text to slug for filename"""
        return text.lower().replace(' ', '_').replace('/', '_')

    def sync_with_external_source(self, source_type: str, endpoint: str) -> Dict[str, Any]:
        """
        Sync KB with external data source
        
        Args:
            source_type: Type of source (api, database, feed, etc.)
            endpoint: Connection endpoint or URL
            
        Returns:
            Sync result
        """
        result = {
            'status': 'pending',
            'source': source_type,
            'timestamp': datetime.now().isoformat(),
            'records_processed': 0,
            'errors': []
        }
        
        try:
            # This would be implemented based on actual data sources
            logger.info(f"Syncing with {source_type} from {endpoint}")
            result['status'] = 'completed'
            
        except Exception as e:
            logger.error(f"Sync error: {e}")
            result['status'] = 'failed'
            result['errors'].append(str(e))
        
        return result

    def get_kb_statistics(self) -> Dict[str, Any]:
        """Get KB statistics"""
        stats = {
            'total_files': 0,
            'total_size_kb': 0,
            'categories': {},
            'last_update': self.metadata.get('last_update'),
            'version': self.metadata.get('version')
        }
        
        try:
            for category_dir in self.kb_path.iterdir():
                if category_dir.is_dir():
                    md_files = list(category_dir.glob('*.md'))
                    total_size = sum(f.stat().st_size for f in md_files) / 1024
                    
                    stats['categories'][category_dir.name] = {
                        'files': len(md_files),
                        'size_kb': round(total_size, 2)
                    }
                    stats['total_files'] += len(md_files)
                    stats['total_size_kb'] += total_size
        except Exception as e:
            logger.error(f"Error calculating statistics: {e}")
        
        stats['total_size_kb'] = round(stats['total_size_kb'], 2)
        return stats

    def validate_kb_integrity(self) -> Dict[str, Any]:
        """Validate KB integrity and completeness"""
        validation_result = {
            'status': 'valid',
            'issues': [],
            'warnings': [],
            'statistics': self.get_kb_statistics()
        }
        
        # Check for required categories
        required_categories = ['equities', 'derivatives', 'ta_indicators', 'fa_basics', 'risk_models', 'strategies']
        for category in required_categories:
            category_path = self.kb_path / category
            if not category_path.exists():
                validation_result['issues'].append(f"Missing category: {category}")
                validation_result['status'] = 'invalid'
        
        # Check for minimum files per category
        for category in required_categories:
            category_path = self.kb_path / category
            if category_path.exists():
                md_files = list(category_path.glob('*.md'))
                if len(md_files) < 2:
                    validation_result['warnings'].append(
                        f"Category {category} has only {len(md_files)} files (recommend >= 2)"
                    )
        
        return validation_result


if __name__ == "__main__":
    config = {
        'kb_path': 'Finance_KB',
        'integration_log_path': 'data/kb_integration.json'
    }
    
    manager = DataIntegrationManager(config)
    
    # Example: Add market data
    market_data = {
        'price': 2500.50,
        'high_52w': 2800,
        'low_52w': 1900,
        'market_cap': '5.2L Cr',
        'beta': 0.95,
        'volatility': 32.5,
        'avg_volume': 5000000,
        'pe_ratio': 28.5,
        'div_yield': 1.2
    }
    manager.add_market_data('RELIANCE', market_data)
    
    # Get statistics
    stats = manager.get_kb_statistics()
    print(f"KB Statistics: {json.dumps(stats, indent=2)}")
    
    # Validate integrity
    validation = manager.validate_kb_integrity()
    print(f"Validation Result: {json.dumps(validation, indent=2)}")

# Finance KB - Quick Start Implementation Guide

## 📋 Overview

This guide walks you through implementing the enhanced Finance Knowledge Base in your project.

## ⚡ Quick Setup (5 minutes)

### Step 1: Install Dependencies
```bash
cd financeKnowlegde
pip install -r requirements.txt
```

### Step 2: Initialize Knowledge Base
```bash
python finance_reasoning/kb_initializer.py
```

**Expected Output:**
```
🚀 FINANCE KB COMPLETE INITIALIZATION & SETUP
📋 Step 1: Pre-Initialization Health Check... ✅ PASS
🔧 Step 2: Building Vectorstore... ✅ PASS
🔗 Step 3: Verifying Integration... ✅ PASS
🏥 Step 4: Post-Initialization Health Check... ✅ PASS
📊 Step 5: Generating Reports... ✅ PASS

✅ Finance KB successfully initialized and ready for production!
```

### Step 3: Verify Installation
```bash
python finance_reasoning/kb_health_checker.py
```

## 🔍 Using the Knowledge Base

### Basic Search
```python
from financeKnowlegde.vectorstore.rag_loader import FinanceRAGLoader

# Initialize
rag = FinanceRAGLoader({
    'kb_path': 'Finance_KB',
    'vectorstore_path': 'vectorstore'
})

# Search
results = rag.retrieve("What are NSE trading hours?", top_k=5)

# Process results
for result in results:
    print(f"Relevance: {result['combined_score']:.3f}")
    print(f"Source: {result['source']}")
    print(f"Content: {result['text'][:200]}...")
    print()
```

### Category-Specific Search
```python
# Search only in derivatives
results = rag.search_by_category(
    "What are options Greeks?",
    category="derivatives",
    top_k=3
)
```

### Find Similar Content
```python
# Get chunks similar to a specific chunk
similar = rag.get_similar_chunks(chunk_id="xyz123", top_k=5)
```

## 📊 Data Integration

### Add Market Data
```python
from financeKnowlegde.finance_reasoning.data_integration_manager import DataIntegrationManager

manager = DataIntegrationManager({
    'kb_path': 'Finance_KB'
})

# Add stock market data
market_data = {
    'price': 2500.50,
    'high_52w': 2800,
    'low_52w': 1900,
    'volatility': 32.5,
    'pe_ratio': 28.5,
    'div_yield': 1.2
}
manager.add_market_data('RELIANCE', market_data)
```

### Add Regulatory Updates
```python
manager.add_regulatory_update(
    title="New SEBI Circuit Breaker Rules",
    content="Effective from January 2025, new circuit breaker mechanisms...",
    update_type="sebi"
)
```

### Add Trading Strategies
```python
manager.add_strategy_update(
    strategy_name="Advanced Mean Reversion",
    content="Rules: Entry when RSI < 30, Exit when RSI > 70...",
    performance={
        'win_rate': 62,
        'profit_factor': 1.8,
        'sharpe_ratio': 1.5,
        'max_drawdown': 12
    }
)
```

## 🏥 Health Monitoring

### Run Full Health Check
```python
from financeKnowlegde.finance_reasoning.kb_health_checker import KBHealthChecker

checker = KBHealthChecker({
    'kb_path': 'Finance_KB'
})

report = checker.run_full_check()
print(f"Status: {report['overall_status']}")
print(f"Files: {report['statistics']['kb_files']}")
print(f"Size: {report['statistics']['kb_size_mb']} MB")
```

### Check Specific Aspects
```python
# Structure validation
structure = checker._check_structure()

# Content quality
quality = checker._check_content_quality()

# Coverage verification
coverage = checker._check_coverage()
```

## 🔗 Integration with MCP Server

### In Your MCP Service
```python
# mcp_service/chat/finance_grounding.py
from financeKnowlegde.vectorstore.rag_loader import FinanceRAGLoader

class FinanceChatGrounding:
    def __init__(self, config):
        self.rag_loader = FinanceRAGLoader(config)
    
    async def ground_response(self, query):
        # Get relevant knowledge
        chunks = self.rag_loader.retrieve(query, top_k=5)
        
        # Build response context
        knowledge_text = "\n".join([c['text'] for c in chunks])
        
        # Generate grounded response
        return {
            'response': your_grounded_response,
            'knowledge_sources': len(chunks),
            'chunks': chunks
        }
```

## ⚙️ Configuration

### Default Configuration
```python
config = {
    'kb_path': 'financeKnowlegde/Finance_KB',
    'vectorstore_path': 'financeKnowlegde/vectorstore',
    'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
    'chunk_size': 512,
    'chunk_overlap': 50,
    'kb_version': '2.0.0'
}
```

### Custom Configuration
```python
# Use larger embedding model for better accuracy
custom_config = {
    'embedding_model': 'sentence-transformers/all-mpnet-base-v2',
    'chunk_size': 256,  # Smaller chunks
    'chunk_overlap': 100
}

rag = FinanceRAGLoader(custom_config)
```

## 📈 Knowledge Base Content

### Available Categories
1. **equities/** - NSE rules, SEBI compliance
2. **derivatives/** - Options, Futures, Greeks
3. **ta_indicators/** - Technical analysis
4. **fa_basics/** - Financial statements
5. **risk_models/** - Risk management
6. **strategies/** - Trading systems
7. **macro/** - Economic indicators
8. **commodities/** - Commodity trading
9. **crypto/** - Cryptocurrency

### Content Size
- **Total Files**: 8-12 per category
- **Total Words**: 40,000+
- **Total Size**: ~500-600 KB
- **Coverage**: Comprehensive financial market knowledge

## 🚀 Production Deployment

### Automated Health Checks
```bash
# Add to crontab for daily checks
0 2 * * * cd /path/to/project && python financeKnowlegde/finance_reasoning/kb_health_checker.py >> logs/kb_health.log
```

### Backup Vectorstore
```bash
# Daily backup
0 3 * * * cp -r financeKnowlegde/vectorstore/ backups/vectorstore_$(date +%Y%m%d)/
```

### Monitor Performance
```python
# Add to your monitoring system
import json

with open('data/kb_health_report.json', 'r') as f:
    report = json.load(f)
    
if report['overall_status'] != 'healthy':
    # Alert team
    send_alert(f"KB health status: {report['overall_status']}")
```

## 🔧 Troubleshooting

### Issue: "Vectorstore not found"
**Solution:**
```bash
python financeKnowlegde/finance_reasoning/kb_initializer.py
```

### Issue: "Low search accuracy"
**Solution:**
```python
# Use larger embedding model
config['embedding_model'] = 'sentence-transformers/all-mpnet-base-v2'

# Rebuild vectorstore
import shutil
shutil.rmtree('financeKnowlegde/vectorstore')
python financeKnowlegde/finance_reasoning/kb_initializer.py
```

### Issue: "Memory usage high"
**Solution:**
```python
# Reduce chunk size
config['chunk_size'] = 256  # Default is 512
config['chunk_overlap'] = 25  # Default is 50
```

## 📚 Additional Resources

- **README.md** - Complete documentation
- **UPGRADE_SUMMARY.md** - What was added
- **data/kb_health_report.json** - Health check results
- **Source code** - Fully documented with docstrings

## ✅ Verification Checklist

After setup, verify:
- [ ] Dependencies installed without errors
- [ ] Initialization completes successfully
- [ ] Health check reports "healthy" status
- [ ] Sample search returns relevant results
- [ ] Integration with your system works
- [ ] Automated health checks scheduled
- [ ] Backups configured

## 🎯 Best Practices

1. **Search Queries**
   - Use specific, domain-relevant terms
   - Example: "NSE circuit breaker levels" (good)
   - Example: "market rules" (too vague)

2. **Content Updates**
   - Update market data monthly
   - Add new strategies quarterly
   - Run full health check annually

3. **Performance**
   - Cache frequent queries
   - Batch queries when possible
   - Monitor search latency

4. **Integration**
   - Handle cases when KB unavailable
   - Implement fallback responses
   - Log KB usage metrics

## 📞 Support

For issues or questions:
1. Check README.md and this guide
2. Review health report: `data/kb_health_report.json`
3. Run health check: `python kb_health_checker.py`
4. Check source code documentation

---

**Status**: ✅ Production Ready  
**Version**: 2.0.0  
**Last Updated**: 2025-01-30

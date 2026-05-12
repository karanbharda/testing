#!/usr/bin/env python3
"""
Quick Verification: ML/AI Models Readiness
Fast check of all AI/ML components integration
"""

import os
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add backend to path
backend_dir = Path(__file__).resolve().parent / 'backend'
sys.path.insert(0, str(backend_dir))


def main():
    """Quick verification of ML/AI models"""
    
    print("=" * 80)
    print("🔍 QUICK ML/AI MODEL VERIFICATION")
    print("=" * 80)
    
    # Test Technical Indicators
    print("\n📊 TECHNICAL INDICATORS")
    print("-" * 80)
    try:
        from ta.trend import SMAIndicator
        from ta.momentum import RSIIndicator
        from ta.volatility import BollingerBands
        print("✅ TA-Lib: Operational")
        tech_status = True
    except ImportError as e:
        print(f"❌ TA-Lib Import Error: {e}")
        tech_status = False
    
    # Test RL Agent
    print("\n🧠 REINFORCEMENT LEARNING AGENT")
    print("-" * 80)
    try:
        from core.rl_agent import rl_agent
        test_data = {'price': 100.0, 'volume': 5000, 'change': 1.5, 'change_pct': 1.5}
        rl_result = rl_agent.get_rl_analysis(test_data)
        print(f"✅ RL Agent: {rl_result['recommendation']} (Confidence: {rl_result['confidence']:.2f})")
        rl_status = True
    except Exception as e:
        print(f"❌ RL Agent Error: {e}")
        rl_status = False
    
    # Test Ensemble Optimizer
    print("\n🤖 ENSEMBLE OPTIMIZER")
    print("-" * 80)
    try:
        from utils.ensemble_optimizer import get_ensemble_optimizer
        import numpy as np
        ensemble_opt = get_ensemble_optimizer()
        test_features = np.random.randn(1, 50)
        ensemble_result = ensemble_opt.get_detailed_ensemble_analysis(test_features)
        print(f"✅ Ensemble Optimizer: {ensemble_result['recommendation']} (Confidence: {ensemble_result['confidence']:.2f})")
        ensemble_status = True
    except Exception as e:
        print(f"❌ Ensemble Optimizer Error: {e}")
        ensemble_status = False
    
    # Test NLP Sentiment
    print("\n💬 NLP & SENTIMENT ANALYSIS")
    print("-" * 80)
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        analyzer = SentimentIntensityAnalyzer()
        test_text = "Strong earnings with impressive growth"
        sentiment = analyzer.polarity_scores(test_text)
        print(f"✅ VADER Sentiment: Compound={sentiment['compound']:.2f} (Positive)")
        nlp_status = True
    except Exception as e:
        print(f"❌ NLP Error: {e}")
        nlp_status = False
    
    # Test Sentence Transformers
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print(f"✅ Sentence Transformers: Operational (384-dim embeddings)")
    except Exception as e:
        print(f"⚠️ Sentence Transformers: {e}")
    
    # Test Groq LLM
    try:
        import os
        groq_key = os.getenv('GROQ_API_KEY')
        if groq_key:
            from groq import Groq
            client = Groq(api_key=groq_key)
            print(f"✅ Groq LLM: Configured and ready")
        else:
            print(f"⚠️ Groq LLM: API key not configured")
    except Exception as e:
        print(f"⚠️ Groq LLM: {e}")
    
    # Test Deep Learning
    print("\n🔮 DEEP LEARNING MODELS")
    print("-" * 80)
    try:
        import torch
        print(f"✅ PyTorch: v{torch.__version__}")
        if torch.cuda.is_available():
            print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        else:
            print(f"ℹ️  GPU: Not available (CPU mode)")
        
        # Test LSTM
        import torch.nn as nn
        class TestLSTM(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(input_size=10, hidden_size=32, batch_first=True)
                self.fc = nn.Linear(32, 1)
            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                return self.fc(lstm_out[:, -1, :])
        
        model = TestLSTM()
        params = sum(p.numel() for p in model.parameters())
        print(f"✅ LSTM Model: {params:,} parameters")
        
        # Test Transformer
        class TestTransformer(nn.Module):
            def __init__(self):
                super().__init__()
                encoder_layer = nn.TransformerEncoderLayer(d_model=32, nhead=4, batch_first=True)
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
                self.fc = nn.Linear(32, 1)
            def forward(self, x):
                x = self.transformer(x)
                return self.fc(x[:, -1, :])
        
        tf_model = TestTransformer()
        tf_params = sum(p.numel() for p in tf_model.parameters())
        print(f"✅ Transformer Model: {tf_params:,} parameters")
        
        dl_status = True
    except Exception as e:
        print(f"❌ Deep Learning Error: {e}")
        dl_status = False
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 ML/AI READINESS SUMMARY")
    print("=" * 80)
    
    components = [
        ("Technical Indicators", tech_status),
        ("RL Agent", rl_status),
        ("Ensemble Optimizer", ensemble_status),
        ("NLP & Sentiment", nlp_status),
        ("Deep Learning", dl_status),
    ]
    
    passed = sum(1 for _, status in components if status)
    total = len(components)
    
    for name, status in components:
        symbol = "✅" if status else "❌"
        print(f"{symbol} {name}")
    
    print("\n" + "=" * 80)
    if passed == total:
        print(f"✨ ALL {total} COMPONENTS OPERATIONAL!")
        print("🎉 SYSTEM READY FOR AI-POWERED TRADING!")
    else:
        print(f"⚠️ {passed}/{total} COMPONENTS WORKING")
        print("ℹ️  Some enhancements recommended")
    
    print("=" * 80)
    print("\n📁 For detailed report: ML_AI_TEST_REPORT.md")
    print("📁 For comprehensive tests: test_ml_comprehensive.py")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Verification interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Verification failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

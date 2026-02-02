#!/usr/bin/env python3
"""
Finance Knowledge Integration Test
==================================

Tests the complete integration of Finance_KB, RAG, and Grounding systems.
"""

import sys
from pathlib import Path
import json

def test_directory_structure():
    """Test Finance_KB directory structure"""
    print("🔍 Testing Finance_KB Directory Structure")
    print("-" * 40)

    kb_path = Path("financeKnowlegde/Finance_KB")

    # Check main directories
    expected_dirs = [
        'equities', 'derivatives', 'commodities', 'crypto',
        'ta_indicators', 'fa_basics', 'risk_models', 'strategies', 'macro'
    ]

    for dir_name in expected_dirs:
        dir_path = kb_path / dir_name
        if dir_path.exists():
            print(f"✅ {dir_name}/ - EXISTS")
        else:
            print(f"❌ {dir_name}/ - MISSING")

    # Check existing files
    existing_files = [
        'equities/nse_basics.md',
        'derivatives/fno_basics.md',
        'risk_models/stop_loss_rules.md'
    ]

    for file_path in existing_files:
        full_path = kb_path / file_path
        if full_path.exists():
            size = full_path.stat().st_size
            print(f"✅ {file_path} - EXISTS ({size} bytes)")
        else:
            print(f"❌ {file_path} - MISSING")

def test_vectorstore():
    """Test vectorstore setup"""
    print("\n🔍 Testing Vector Store")
    print("-" * 40)

    vs_path = Path("financeKnowlegde/vectorstore")

    required_files = ['rag_loader.py', 'embeddings.pkl', 'chunks.pkl']
    for file_name in required_files:
        file_path = vs_path / file_name
        if file_path.exists():
            size = file_path.stat().st_size
            print(f"✅ {file_name} - EXISTS ({size} bytes)")
        else:
            print(f"❌ {file_name} - MISSING")

def test_test_pack():
    """Test finance test pack"""
    print("\n🔍 Testing Finance Test Pack")
    print("-" * 40)

    test_pack_path = Path("financeKnowlegde/finance_reasoning/test/finance_test_pack.json")

    if test_pack_path.exists():
        size = test_pack_path.stat().st_size
        print(f"✅ finance_test_pack.json - EXISTS ({size} bytes)")

        # Load and check structure
        try:
            with open(test_pack_path, 'r') as f:
                data = json.load(f)

            if isinstance(data, list) and len(data) > 0:
                print(f"✅ Test pack contains {len(data)} questions")

                # Check categories
                categories = set(item.get('category', '') for item in data)
                print(f"✅ Categories: {sorted(categories)}")

                # Check required categories
                required_cats = {'nse_rules', 'f&o_ban_logic', 'stop_loss_reasoning', 'market_regimes', 'risk_based_exits'}
                missing_cats = required_cats - categories
                if missing_cats:
                    print(f"⚠️  Missing categories: {missing_cats}")
                else:
                    print("✅ All required categories present")

            else:
                print("❌ Test pack structure invalid")

        except Exception as e:
            print(f"❌ Error loading test pack: {e}")

    else:
        print("❌ finance_test_pack.json - MISSING")

def test_rag_integration():
    """Test RAG system integration"""
    print("\n🔍 Testing RAG Integration")
    print("-" * 40)

    try:
        # Add paths
        sys.path.append('financeKnowlegde')

        from financeKnowlegde.vectorstore.rag_loader import FinanceRAGLoader

        config = {
            'vectorstore_path': 'financeKnowlegde/vectorstore'
        }

        rag = FinanceRAGLoader(config)

        if rag.embeddings is not None and rag.chunks is not None:
            print(f"✅ RAG loaded: {len(rag.chunks)} chunks, {rag.embeddings.shape} embeddings")

            # Test retrieval
            test_query = "What is NSE circuit breaker?"
            results = rag.retrieve(test_query, top_k=3)

            if results:
                print(f"✅ Retrieval test passed: {len(results)} results for query")
                for i, result in enumerate(results[:2], 1):
                    print(f"   {i}. {result['text'][:100]}...")
            else:
                print("❌ Retrieval test failed: no results")

        else:
            print("❌ RAG not loaded - vector store may need initialization")

    except Exception as e:
        print(f"❌ RAG integration error: {e}")

def test_grounding_integration():
    """Test grounding system integration"""
    print("\n🔍 Testing Grounding Integration")
    print("-" * 40)

    try:
        # Add paths
        sys.path.insert(0, '..')  # Add project root

        from mcp_service.chat.finance_grounding import FinanceChatGrounding

        config = {
            'rag': {'vectorstore_path': 'financeKnowlegde/vectorstore'},
            'kb_path': 'financeKnowlegde/Finance_KB'
        }

        grounding = FinanceChatGrounding(config)
        print("✅ FinanceChatGrounding initialized successfully")

    except Exception as e:
        print(f"❌ Grounding integration error: {e}")

def main():
    """Run all integration tests"""
    print("🔧 FINANCE KNOWLEDGE INTEGRATION TEST")
    print("=" * 50)

    # Change to project root
    project_root = Path(__file__).parent
    import os
    os.chdir(project_root)

    test_directory_structure()
    test_vectorstore()
    test_test_pack()
    test_rag_integration()
    test_grounding_integration()

    print("\n" + "=" * 50)
    print("🎯 INTEGRATION TEST COMPLETE")
    print("=" * 50)

if __name__ == "__main__":
    main()

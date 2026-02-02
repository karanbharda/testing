#!/usr/bin/env python3
"""
Finance KB Vector Store Initializer
===================================

Initializes the RAG vector store with Finance_KB content.
Processes all markdown files and creates deterministic embeddings.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from vectorstore.rag_loader import FinanceRAGLoader

def main():
    """Initialize finance KB vector store"""
    print("🔧 FINANCE KB VECTOR STORE INITIALIZATION")
    print("=" * 50)

    # Configuration
    config = {
        'rag': {
            'vectorstore_path': 'vectorstore',
            'kb_path': 'Finance_KB'
        }
    }

    # Initialize RAG loader
    rag_loader = FinanceRAGLoader(config)

    # Check if KB exists
    kb_path = Path(config['rag']['kb_path'])
    if not kb_path.exists():
        print(f"❌ Finance_KB directory not found at {kb_path}")
        return 1

    # Find all markdown files
    markdown_files = list(kb_path.rglob("*.md"))
    if not markdown_files:
        print("❌ No markdown files found in Finance_KB")
        return 1

    print(f"📁 Found {len(markdown_files)} markdown files:")
    for file in markdown_files:
        print(f"   • {file.relative_to(kb_path)}")

    # Initialize vector store
    print("\n🔄 Initializing vector store...")
    try:
        rag_loader.build_vectorstore(config['rag']['kb_path'])

        # Check if vector store was created
        if rag_loader.embeddings is not None and rag_loader.chunks is not None:
            print("✅ Vector store initialized successfully")
            print(f"📊 Vector store saved to: {config['rag']['vectorstore_path']}")
            print(f"📈 Chunks: {len(rag_loader.chunks)}, Dimensions: {rag_loader.embeddings.shape[1]}")
            return 0
        else:
            print("❌ Vector store initialization failed - no data loaded")
            return 1

    except Exception as e:
        print(f"❌ Vector store initialization failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
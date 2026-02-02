#!/usr/bin/env python3
"""
RAG Loader for Finance Knowledge Base
=====================================

Loads and manages vector embeddings for deterministic retrieval.
Single vector store with versioned chunks.
Uses sentence-transformers for semantic embeddings.
"""

import os
import pickle
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np
from datetime import datetime
import logging
from collections import Counter

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

logger = logging.getLogger(__name__)

class FinanceRAGLoader:
    """
    RAG Loader for Finance Knowledge Base

    Provides deterministic retrieval from single vector store.
    No experimentation - production-ready implementation.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize RAG Loader

        Args:
            config: Configuration with vectorstore_path, embedding_model, etc.
        """
        self.config = config
        self.vectorstore_path = Path(config.get("vectorstore_path", "vectorstore"))
        self.embeddings_file = self.vectorstore_path / "embeddings.pkl"
        self.chunks_file = self.vectorstore_path / "chunks.pkl"

        # Embedding parameters
        self.embedding_model_name = config.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
        self.embedding_model = None  # Lazy load
        self.chunk_size = config.get("chunk_size", 512)
        self.chunk_overlap = config.get("chunk_overlap", 50)

        # Version control
        self.version = config.get("kb_version", "2.0.0")

        # Initialize storage
        self.embeddings = None
        self.chunks = None
        self.metadata = None

        # Initialize embedding model
        self._init_embedding_model()
        self._load_vectorstore()

    def _init_embedding_model(self):
        """Initialize the embedding model"""
        if HAS_SENTENCE_TRANSFORMERS:
            try:
                self.embedding_model = SentenceTransformer(self.embedding_model_name)
                logger.info(f"Loaded embedding model: {self.embedding_model_name}")
            except Exception as e:
                logger.warning(f"Failed to load sentence-transformers model: {e}. Using fallback.")
                self.embedding_model = None
        else:
            logger.warning("sentence-transformers not installed. Using fallback embedding method.")

    def _load_vectorstore(self):
        """Load vector store from disk"""
        try:
            if self.embeddings_file.exists() and self.chunks_file.exists():
                with open(self.embeddings_file, 'rb') as f:
                    self.embeddings = pickle.load(f)

                with open(self.chunks_file, 'rb') as f:
                    chunk_data = pickle.load(f)
                    self.chunks = chunk_data['chunks']
                    self.metadata = chunk_data['metadata']

                logger.info(f"Loaded vector store with {len(self.chunks)} chunks")
            else:
                logger.warning("Vector store not found, needs initialization")
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")

    def _save_vectorstore(self):
        """Save vector store to disk"""
        try:
            self.vectorstore_path.mkdir(exist_ok=True)

            with open(self.embeddings_file, 'wb') as f:
                pickle.dump(self.embeddings, f)

            chunk_data = {
                'chunks': self.chunks,
                'metadata': self.metadata,
                'version': self.version,
                'created': datetime.now().isoformat()
            }

            with open(self.chunks_file, 'wb') as f:
                pickle.dump(chunk_data, f)

            logger.info(f"Saved vector store with {len(self.chunks)} chunks")
        except Exception as e:
            logger.error(f"Error saving vector store: {e}")

    def _chunk_text(self, text: str, source: str) -> List[Dict[str, Any]]:
        """Chunk text into manageable pieces with metadata"""
        words = text.split()
        chunks = []

        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = ' '.join(chunk_words)

            if len(chunk_words) >= 50:  # Minimum chunk size
                chunk_id = hashlib.md5(f"{source}_{i}_{chunk_text}".encode()).hexdigest()[:16]

                chunk = {
                    'id': chunk_id,
                    'text': chunk_text,
                    'source': source,
                    'chunk_index': i,
                    'word_count': len(chunk_words),
                    'version': self.version,
                    'created': datetime.now().isoformat()
                }
                chunks.append(chunk)

        return chunks

    def _generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate semantic embedding for text
        
        Uses sentence-transformers if available, fallback to TF-IDF style embedding
        """
        if self.embedding_model is not None:
            # Use sentence-transformers for semantic embeddings
            try:
                embedding = self.embedding_model.encode(text, convert_to_numpy=True)
                return embedding.astype(np.float32)
            except Exception as e:
                logger.warning(f"Error generating embedding with sentence-transformers: {e}")
        
        # Fallback: Create embedding from text statistics and keywords
        # This is deterministic and works without external models
        embedding = self._fallback_embedding(text)
        return embedding.astype(np.float32)
    
    def _fallback_embedding(self, text: str, dim: int = 384) -> np.ndarray:
        """
        Fallback embedding method using text statistics
        Creates a deterministic embedding from text features
        """
        # Financial keywords for boost
        financial_keywords = {
            'price': 1.0, 'volume': 1.0, 'risk': 1.0, 'return': 1.0,
            'volatility': 1.0, 'margin': 1.0, 'support': 0.9, 'resistance': 0.9,
            'trend': 0.9, 'momentum': 0.9, 'rsi': 0.8, 'macd': 0.8,
            'moving average': 0.8, 'dividend': 0.8, 'earnings': 0.8,
            'balance sheet': 0.8, 'cash flow': 0.8, 'ratio': 0.7,
            'nifty': 1.0, 'nse': 1.0, 'sebi': 1.0, 'trading': 0.9,
            'option': 0.9, 'future': 0.9, 'derivative': 0.9, 'settlement': 0.8,
            'hedging': 0.8, 'breakout': 0.8, 'reversal': 0.8, 'entry': 0.7,
            'exit': 0.7, 'stop loss': 0.9, 'profit': 0.8, 'loss': 0.7
        }
        
        # Initialize embedding
        embedding = np.zeros(dim, dtype=np.float32)
        
        # Character-level hash for determinism
        text_lower = text.lower()
        hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
        
        # Distribute hash across embedding
        for i in range(dim):
            seed = (hash_val + i * 31) % (2**32)
            embedding[i] = ((seed ^ (i * 7)) % 256) / 255.0
        
        # Boost dimensions for financial keywords
        word_tokens = text_lower.split()
        keyword_boost = np.zeros(dim, dtype=np.float32)
        
        for keyword, weight in financial_keywords.items():
            if keyword in text_lower:
                # Boost several dimensions based on keyword
                keyword_hash = int(hashlib.md5(keyword.encode()).hexdigest(), 16)
                for j in range(5):  # Boost 5 dimensions per keyword
                    idx = (keyword_hash + j) % dim
                    keyword_boost[idx] += weight * 0.1
        
        # Add keyword boosts
        embedding += keyword_boost
        
        # Add statistical features
        text_len = len(text_lower)
        word_count = len(word_tokens)
        
        # Length-based features
        embedding[0:10] += (text_len % 256) / 256.0
        embedding[10:20] += (word_count % 256) / 256.0
        
        # Character frequency features
        char_counts = Counter(text_lower)
        unique_chars = len(char_counts)
        embedding[20:30] += (unique_chars % 256) / 256.0
        
        # Normalize to unit vector
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding.astype(np.float32)

    def build_vectorstore(self, kb_path: str):
        """Build vector store from knowledge base"""
        kb_path = Path(kb_path)
        all_chunks = []
        all_embeddings = []

        logger.info(f"Building vector store from {kb_path}")

        # Process all markdown files
        for md_file in kb_path.rglob("*.md"):
            try:
                with open(md_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Chunk the content
                chunks = self._chunk_text(content, str(md_file.relative_to(kb_path)))

                for chunk in chunks:
                    embedding = self._generate_embedding(chunk['text'])

                    all_chunks.append(chunk)
                    all_embeddings.append(embedding)

                logger.info(f"Processed {md_file}: {len(chunks)} chunks")

            except Exception as e:
                logger.error(f"Error processing {md_file}: {e}")

        # Convert to numpy arrays
        self.embeddings = np.array(all_embeddings, dtype=np.float32)
        self.chunks = all_chunks

        # Build metadata
        self.metadata = {
            'total_chunks': len(all_chunks),
            'embedding_dim': self.embeddings.shape[1] if len(all_embeddings) > 0 else 0,
            'kb_version': self.version,
            'build_date': datetime.now().isoformat(),
            'sources': list(set(chunk['source'] for chunk in all_chunks))
        }

        # Save to disk
        self._save_vectorstore()

        logger.info(f"Built vector store: {len(all_chunks)} chunks, {self.embeddings.shape}")

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve most relevant chunks for query with multi-factor ranking

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of relevant chunks with scores
        """
        if self.embeddings is None or self.chunks is None:
            logger.warning("Vector store not loaded")
            return []

        # Generate query embedding
        query_embedding = self._generate_embedding(query)

        # Calculate cosine similarities
        similarities = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding) + 1e-8
        )

        # Multi-factor ranking
        scores = self._compute_ranking_scores(query, similarities)

        # Get top-k indices by combined score
        top_indices = np.argsort(scores)[::-1][:top_k]

        # Build results
        results = []
        for idx in top_indices:
            chunk = self.chunks[idx].copy()
            chunk['semantic_score'] = float(similarities[idx])
            chunk['combined_score'] = float(scores[idx])
            results.append(chunk)

        return results

    def _compute_ranking_scores(self, query: str, semantic_scores: np.ndarray) -> np.ndarray:
        """
        Compute combined ranking scores using multiple factors
        
        Factors:
        1. Semantic similarity (80%)
        2. Keyword matching (15%)
        3. Chunk quality (5%)
        """
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        # Initialize scores
        combined_scores = semantic_scores.copy() * 0.80
        
        # Keyword matching bonus
        keyword_scores = np.zeros(len(self.chunks), dtype=np.float32)
        for i, chunk in enumerate(self.chunks):
            text_lower = chunk['text'].lower()
            chunk_words = set(text_lower.split())
            
            # Exact word matches
            matches = len(query_words & chunk_words)
            max_matches = max(len(query_words), 1)
            match_score = min(matches / max_matches, 1.0)
            
            # Substring matching for phrases
            if len(query) > 5 and query_lower in text_lower:
                match_score = min(match_score + 0.3, 1.0)
            
            keyword_scores[i] = match_score
        
        combined_scores += keyword_scores * 0.15
        
        # Chunk quality bonus (prefer longer, more complete chunks)
        quality_scores = np.zeros(len(self.chunks), dtype=np.float32)
        for i, chunk in enumerate(self.chunks):
            word_count = chunk.get('word_count', 50)
            # Prefer chunks with 100-500 words
            if word_count >= 100:
                quality_scores[i] = min(word_count / 500, 1.0)
            else:
                quality_scores[i] = word_count / 100 * 0.5
        
        combined_scores += quality_scores * 0.05
        
        return combined_scores
    
    def search_by_category(self, query: str, category: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search within specific knowledge category
        
        Args:
            query: Search query
            category: KB category (equities, derivatives, ta_indicators, etc.)
            top_k: Number of results
            
        Returns:
            Filtered results from specified category
        """
        all_results = self.retrieve(query, top_k * 3)  # Get more to filter
        
        # Filter by category
        category_lower = category.lower()
        filtered = [
            r for r in all_results 
            if category_lower in r['source'].lower()
        ]
        
        return filtered[:top_k]
    
    def get_similar_chunks(self, chunk_id: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Get similar chunks to a given chunk
        
        Args:
            chunk_id: ID of chunk to find similar to
            top_k: Number of similar chunks
            
        Returns:
            Similar chunks
        """
        # Find the chunk
        chunk_idx = None
        for i, chunk in enumerate(self.chunks):
            if chunk['id'] == chunk_id:
                chunk_idx = i
                break
        
        if chunk_idx is None:
            logger.warning(f"Chunk {chunk_id} not found")
            return []
        
        # Find similar embeddings
        query_embedding = self.embeddings[chunk_idx]
        similarities = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding) + 1e-8
        )
        
        # Get top-k (excluding self)
        indices = np.argsort(similarities)[::-1]
        results = []
        for idx in indices:
            if idx != chunk_idx and len(results) < top_k:
                chunk = self.chunks[idx].copy()
                chunk['score'] = float(similarities[idx])
                results.append(chunk)
        
        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        if self.metadata:
            return self.metadata

        return {
            'status': 'not_loaded',
            'chunks': len(self.chunks) if self.chunks else 0,
            'embedding_dim': self.embeddings.shape[1] if self.embeddings is not None else 0
        }
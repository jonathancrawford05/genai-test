"""
Hybrid retriever combining semantic (ONNX) and keyword (BM25) search.

Provides better retrieval for diverse query types:
- Semantic search: Good for conceptual/reasoning queries
- BM25: Good for keyword matching, enumerations, exact terms
- Hybrid: Best of both worlds
"""
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
import re


class HybridRetriever:
    """
    Combines semantic embeddings with BM25 keyword search.

    Strategy:
    - Retrieve from both semantic and BM25 indexes
    - Merge results with configurable weighting
    - Deduplicate and rank

    Usage:
        hybrid = HybridRetriever(onnx_processor)
        hybrid.build_bm25_index()
        results = hybrid.search("list all rating plan rules", alpha=0.7)
    """

    def __init__(self, processor):
        """
        Initialize hybrid retriever.

        Args:
            processor: ONNXProcessor instance with semantic index
        """
        self.semantic_processor = processor
        self.bm25_index = None
        self.corpus_docs = []  # Full document texts
        self.corpus_metadata = []  # Corresponding metadata
        self.corpus_ids = []  # Document IDs for deduplication

    def build_bm25_index(self, verbose: bool = False):
        """
        Build BM25 index from all chunks in ChromaDB.

        Args:
            verbose: Print build progress
        """
        if verbose:
            print("Building BM25 index from ChromaDB...")

        # Get all chunks from ChromaDB
        collection_data = self.semantic_processor.collection.get(
            include=['documents', 'metadatas']
        )

        if not collection_data or not collection_data.get('documents'):
            raise ValueError("No documents found in ChromaDB collection")

        self.corpus_docs = collection_data['documents']
        self.corpus_metadata = collection_data['metadatas']

        # Generate IDs for deduplication (source_file + chunk_index)
        self.corpus_ids = [
            f"{meta['source_file']}_{meta['chunk_index']}"
            for meta in self.corpus_metadata
        ]

        # Tokenize corpus for BM25
        tokenized_corpus = [self._tokenize(doc) for doc in self.corpus_docs]

        # Build BM25 index
        self.bm25_index = BM25Okapi(tokenized_corpus)

        if verbose:
            print(f"✓ BM25 index built with {len(self.corpus_docs)} documents")

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text for BM25.

        Simple approach: lowercase + split on whitespace/punctuation.

        Args:
            text: Text to tokenize

        Returns:
            List of tokens
        """
        # Lowercase
        text = text.lower()

        # Split on whitespace and punctuation, keep alphanumeric + hyphens
        tokens = re.findall(r'\b[\w\-]+\b', text)

        return tokens

    def search(
        self,
        query: str,
        top_k: int = 5,
        alpha: float = 0.5,
        where: Optional[Dict] = None,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Hybrid search combining semantic + BM25.

        Args:
            query: Search query
            top_k: Number of results to return
            alpha: Weight for semantic search (1-alpha for BM25)
                   alpha=1.0: Pure semantic
                   alpha=0.0: Pure BM25
                   alpha=0.5: Balanced
            where: Optional ChromaDB where clause (for pre-filtering)
            verbose: Print search details

        Returns:
            ChromaDB-style results dict with merged rankings
        """
        if self.bm25_index is None:
            raise ValueError("BM25 index not built. Call build_bm25_index() first.")

        # 1. Semantic search
        semantic_results = self._semantic_search(query, top_k * 2, where, verbose)

        # 2. BM25 search
        bm25_results = self._bm25_search(query, top_k * 2, where, verbose)

        # 3. Merge with RRF (Reciprocal Rank Fusion)
        merged_results = self._rrf_merge(
            semantic_results,
            bm25_results,
            alpha,
            top_k,
            verbose
        )

        return merged_results

    def _semantic_search(
        self,
        query: str,
        top_k: int,
        where: Optional[Dict],
        verbose: bool
    ) -> List[Dict]:
        """Perform semantic search via ChromaDB."""
        try:
            if where:
                # Pre-filtered search
                results = self.semantic_processor.collection.query(
                    query_texts=[query],
                    n_results=top_k,
                    where=where
                )
            else:
                # Regular search
                results = self.semantic_processor.query(query, top_k=top_k)

            # Convert to list of dicts with rankings
            semantic_list = []
            if results and results.get('documents') and results['documents'][0]:
                for rank, (doc, meta, dist) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results.get('distances', [[]])[0] if results.get('distances') else [None] * len(results['documents'][0])
                )):
                    doc_id = f"{meta['source_file']}_{meta['chunk_index']}"
                    semantic_list.append({
                        'id': doc_id,
                        'text': doc,
                        'metadata': meta,
                        'rank': rank,
                        'distance': dist
                    })

            if verbose:
                print(f"  Semantic: {len(semantic_list)} results")

            return semantic_list

        except Exception as e:
            if verbose:
                print(f"  ⚠️  Semantic search failed: {e}")
            return []

    def _bm25_search(
        self,
        query: str,
        top_k: int,
        where: Optional[Dict],
        verbose: bool
    ) -> List[Dict]:
        """Perform BM25 keyword search."""
        # Tokenize query
        query_tokens = self._tokenize(query)

        # Get BM25 scores for all documents
        scores = self.bm25_index.get_scores(query_tokens)

        # Get top-k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k * 3]

        # Filter by where clause if provided
        bm25_results = []
        for idx in top_indices:
            if len(bm25_results) >= top_k:
                break

            meta = self.corpus_metadata[idx]

            # Apply where clause filtering
            if where:
                if not self._matches_where(meta, where):
                    continue

            doc_id = self.corpus_ids[idx]
            bm25_results.append({
                'id': doc_id,
                'text': self.corpus_docs[idx],
                'metadata': meta,
                'rank': len(bm25_results),
                'score': scores[idx]
            })

        if verbose:
            print(f"  BM25: {len(bm25_results)} results")

        return bm25_results

    def _matches_where(self, metadata: Dict, where: Dict) -> bool:
        """
        Check if metadata matches ChromaDB where clause.

        Simplified implementation for common cases.
        """
        if "$and" in where:
            return all(self._matches_where(metadata, cond) for cond in where["$and"])

        if "$in" in where.get("source_file", {}):
            return metadata.get("source_file") in where["source_file"]["$in"]

        # Direct equality
        for key, value in where.items():
            if key.startswith("$"):
                continue
            if metadata.get(key) != value:
                return False

        return True

    def _rrf_merge(
        self,
        semantic_results: List[Dict],
        bm25_results: List[Dict],
        alpha: float,
        top_k: int,
        verbose: bool
    ) -> Dict[str, Any]:
        """
        Merge results using Reciprocal Rank Fusion (RRF).

        RRF formula: score = alpha * (1/(k + semantic_rank)) + (1-alpha) * (1/(k + bm25_rank))
        where k=60 (standard constant)

        Args:
            semantic_results: Results from semantic search
            bm25_results: Results from BM25 search
            alpha: Weight for semantic (1-alpha for BM25)
            top_k: Final number of results
            verbose: Print merge details

        Returns:
            ChromaDB-style results dict
        """
        k = 60  # RRF constant

        # Build lookup maps
        semantic_ranks = {r['id']: r['rank'] for r in semantic_results}
        bm25_ranks = {r['id']: r['rank'] for r in bm25_results}

        # Get all unique document IDs
        all_ids = set(semantic_ranks.keys()) | set(bm25_ranks.keys())

        # Calculate RRF scores
        rrf_scores = {}
        doc_data = {}

        for doc_id in all_ids:
            semantic_rank = semantic_ranks.get(doc_id, 9999)
            bm25_rank = bm25_ranks.get(doc_id, 9999)

            # RRF score with alpha weighting
            rrf_score = (
                alpha * (1.0 / (k + semantic_rank)) +
                (1.0 - alpha) * (1.0 / (k + bm25_rank))
            )

            rrf_scores[doc_id] = rrf_score

            # Store document data (prefer from semantic if available)
            if doc_id in semantic_ranks:
                doc_data[doc_id] = next(r for r in semantic_results if r['id'] == doc_id)
            else:
                doc_data[doc_id] = next(r for r in bm25_results if r['id'] == doc_id)

        # Sort by RRF score
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:top_k]

        # Build ChromaDB-style results
        merged_results = {
            'documents': [[doc_data[doc_id]['text'] for doc_id in sorted_ids]],
            'metadatas': [[doc_data[doc_id]['metadata'] for doc_id in sorted_ids]],
            'distances': [[1.0 - rrf_scores[doc_id] for doc_id in sorted_ids]]  # Convert score to distance
        }

        if verbose:
            print(f"  Merged: {len(sorted_ids)} results (alpha={alpha:.2f})")
            if sorted_ids:
                top_id = sorted_ids[0]
                sem_rank = semantic_ranks.get(top_id, "N/A")
                bm25_rank = bm25_ranks.get(top_id, "N/A")
                print(f"    Top result: sem_rank={sem_rank}, bm25_rank={bm25_rank}, rrf={rrf_scores[top_id]:.4f}")

        return merged_results

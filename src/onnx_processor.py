"""
PDF processor using ChromaDB's default ONNX embeddings (all-MiniLM-L6-v2).
"""
from typing import List, Dict

from src.base_processor import BasePDFProcessor


class ONNXProcessor(BasePDFProcessor):
    """
    Memory-efficient processor using ChromaDB's default ONNX embedding function.

    - Model: all-MiniLM-L6-v2
    - Size: 79MB
    - Dimensions: 384
    - Runtime: ONNX (CPU-optimized)
    - Memory: ~500MB peak
    """

    def _setup_collection(self):
        """
        Create collection with ChromaDB's default ONNX embedding function.
        Validates dimension compatibility and recreates if needed.
        """
        collection_name = self.collection_name

        # Try to get existing collection
        try:
            collection = self.client.get_collection(name=collection_name)

            # Check if dimensions match (ONNX = 384)
            metadata = collection.metadata or {}
            stored_dims = metadata.get("dimensions")

            if stored_dims and stored_dims != 384:
                print(f"⚠️  Collection has wrong dimensions ({stored_dims} != 384)")
                print(f"   Deleting and recreating collection...")
                self.client.delete_collection(name=collection_name)
                collection = None
        except Exception:
            # Collection doesn't exist
            collection = None

        # Create if doesn't exist
        if collection is None:
            collection = self.client.create_collection(
                name=collection_name,
                metadata={
                    "description": "PDF chunks with ONNX embeddings",
                    "embedding_model": "all-MiniLM-L6-v2",
                    "embedding_type": "onnx",
                    "dimensions": 384,
                },
            )

        print(f"✓ ONNX collection ready: {collection.count()} existing chunks")
        return collection

    def _add_batch(self, texts: List[str], metadatas: List[Dict], ids: List[str]):
        """
        Add batch to ChromaDB (embeddings generated automatically).

        Args:
            texts: List of document texts
            metadatas: List of metadata dicts
            ids: List of unique IDs
        """
        if not texts:
            return

        try:
            # ChromaDB auto-generates embeddings with default function
            self.collection.add(
                documents=texts,
                metadatas=metadatas,
                ids=ids,
            )
        except Exception as e:
            print(f"\n    Warning: Batch add failed: {e}")

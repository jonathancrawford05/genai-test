"""
PDF processor using Ollama embeddings (nomic-embed-text).
"""
from typing import List, Dict

import ollama

from src.base_processor import BasePDFProcessor


class OllamaProcessor(BasePDFProcessor):
    """
    RAG processor using Ollama's nomic-embed-text model.

    - Model: nomic-embed-text
    - Size: 274MB
    - Dimensions: 768
    - Runtime: Ollama API
    - Memory: ~800MB-1GB peak
    - Training: RAG-optimized
    """

    def __init__(
        self,
        persist_directory: str = "./chroma_db_ollama",
        collection_name: str = "pdf_ollama_embeddings",
        model_name: str = "nomic-embed-text",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        batch_size: int = 20,
    ):
        """
        Initialize with Ollama embedding model.

        Args:
            persist_directory: ChromaDB persistence directory
            collection_name: Collection name
            model_name: Ollama embedding model
            chunk_size: Characters per chunk
            chunk_overlap: Overlap between chunks
            batch_size: Chunks per batch
        """
        self.model_name = model_name

        # Test Ollama connection and get embedding dimension
        print(f"Testing Ollama connection with {model_name}...")
        try:
            test_embedding = ollama.embeddings(model=model_name, prompt="test")
            self.embedding_dim = len(test_embedding['embedding'])
            print(f"✓ Ollama connected (dim={self.embedding_dim})")
        except Exception as e:
            raise RuntimeError(
                f"Failed to connect to Ollama: {e}\n"
                f"Make sure:\n"
                f"  1. Ollama is running (ollama serve)\n"
                f"  2. Model is installed (ollama pull {model_name})"
            )

        # Initialize base class (will call _setup_collection)
        super().__init__(
            persist_directory=persist_directory,
            collection_name=collection_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            batch_size=batch_size,
        )

    def _setup_collection(self):
        """
        Create collection for Ollama embeddings.
        Validates dimension compatibility and recreates if needed.
        """
        collection_name = self.collection_name

        # Try to get existing collection
        try:
            collection = self.client.get_collection(name=collection_name)

            # Check if dimensions match (should be same as self.embedding_dim)
            metadata = collection.metadata or {}
            stored_dims = metadata.get("dimensions")

            if stored_dims and stored_dims != self.embedding_dim:
                print(f"⚠️  Collection has wrong dimensions ({stored_dims} != {self.embedding_dim})")
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
                    "description": "PDF chunks with Ollama embeddings",
                    "embedding_model": self.model_name,
                    "embedding_type": "ollama",
                    "dimensions": self.embedding_dim,
                },
            )

        print(f"✓ Ollama collection ready: {collection.count()} existing chunks")
        return collection

    def _add_batch(self, texts: List[str], metadatas: List[Dict], ids: List[str]):
        """
        Generate embeddings via Ollama and add batch to ChromaDB.

        Args:
            texts: List of document texts
            metadatas: List of metadata dicts
            ids: List of unique IDs
        """
        if not texts:
            return

        try:
            # Generate embeddings for all texts in batch
            embeddings = []
            for text in texts:
                response = ollama.embeddings(model=self.model_name, prompt=text)
                embeddings.append(response['embedding'])

            # Add batch with pre-computed embeddings
            self.collection.add(
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids,
            )

        except Exception as e:
            print(f"\n    Warning: Batch add failed: {e}")

    def query(self, query_text: str, top_k: int = 5):
        """
        Query collection with Ollama-generated embedding.

        Must override base class because Ollama collections require
        embeddings to be generated via Ollama API, not ChromaDB default.

        Args:
            query_text: Query string
            top_k: Number of results

        Returns:
            Query results from ChromaDB
        """
        # Generate embedding for query using Ollama
        response = ollama.embeddings(model=self.model_name, prompt=query_text)
        query_embedding = response['embedding']

        # Query with pre-computed embedding
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )
        return results


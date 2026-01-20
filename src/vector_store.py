"""
ChromaDB vector store with memory-efficient operations.
"""
from typing import List, Optional
import gc

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from .pdf_processor import DocumentChunk


class VectorStore:
    """Manages ChromaDB vector store with embedding generation."""

    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        collection_name: str = "pdf_documents",
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        """
        Initialize vector store.

        Args:
            persist_directory: Directory to persist ChromaDB data
            collection_name: Name of the collection
            embedding_model: SentenceTransformer model name
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model

        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
            ),
        )

        # Load embedding model (lightweight, runs on CPU)
        print(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        print(f"Embedding dimension: {self.embedding_model.get_sentence_embedding_dimension()}")

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "PDF document chunks with embeddings"},
        )

    def add_chunks(self, chunks: List[DocumentChunk], batch_size: int = 50) -> int:
        """
        Add chunks to the vector store in batches.

        Args:
            chunks: List of DocumentChunk objects
            batch_size: Number of chunks to process at once

        Returns:
            Number of chunks added
        """
        if not chunks:
            return 0

        total_added = 0

        # Process in smaller batches to control memory
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]

            # Extract texts and metadata
            texts = [chunk.text for chunk in batch]
            metadatas = [chunk.metadata for chunk in batch]

            # Generate embeddings
            embeddings = self.embedding_model.encode(
                texts,
                show_progress_bar=False,
                convert_to_numpy=True,
            ).tolist()

            # Generate IDs
            ids = [
                f"{chunk.metadata['source_file']}_p{chunk.metadata['page_number']}_{i + j}"
                for j, chunk in enumerate(batch)
            ]

            # Add to collection
            self.collection.add(
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids,
            )

            total_added += len(batch)

            # Clean up
            del embeddings
            gc.collect()

        return total_added

    def query(
        self, query_text: str, top_k: int = 5, filter_metadata: Optional[dict] = None
    ) -> List[dict]:
        """
        Query the vector store.

        Args:
            query_text: Query string
            top_k: Number of results to return
            filter_metadata: Optional metadata filter

        Returns:
            List of results with documents, metadata, and distances
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode(
            [query_text],
            show_progress_bar=False,
            convert_to_numpy=True,
        ).tolist()[0]

        # Query collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filter_metadata,
        )

        # Format results
        formatted_results = []
        for i in range(len(results["documents"][0])):
            formatted_results.append(
                {
                    "document": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i],
                }
            )

        return formatted_results

    def count(self) -> int:
        """Get total number of chunks in the store."""
        return self.collection.count()

    def clear(self):
        """Clear all data from the collection."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"description": "PDF document chunks with embeddings"},
        )

    def get_stats(self) -> dict:
        """Get statistics about the vector store."""
        return {
            "collection_name": self.collection_name,
            "total_chunks": self.count(),
            "persist_directory": self.persist_directory,
            "embedding_model": self.embedding_model_name,
            "embedding_dimension": self.embedding_model.get_sentence_embedding_dimension(),
        }

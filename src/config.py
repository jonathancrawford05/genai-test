"""
Configuration management for memory-efficient RAG pipeline.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import os


@dataclass
class PDFProcessingConfig:
    """Configuration for PDF processing."""

    # Batch processing
    batch_size: int = 1  # Process N PDFs at a time to limit memory (1 = safest)
    chunk_size: int = 1000  # Characters per chunk
    chunk_overlap: int = 200  # Overlap between chunks

    # Memory management
    max_chunks_in_memory: int = 50  # Max chunks to hold before flushing to DB

    # Text extraction
    extract_tables: bool = True
    extract_images: bool = False


@dataclass
class VectorStoreConfig:
    """Configuration for ChromaDB vector store."""

    # Database settings
    persist_directory: str = "./chroma_db"
    collection_name: str = "pdf_documents"

    # Embedding model
    embedding_model: str = "all-MiniLM-L6-v2"  # Fast, lightweight model (384 dims)

    # Search settings
    top_k: int = 5  # Number of chunks to retrieve

    def __post_init__(self):
        """Ensure persist directory exists."""
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)


@dataclass
class RAGConfig:
    """Configuration for RAG query engine."""

    # LLM settings
    model_provider: str = "anthropic"  # or "openai"
    model_name: str = "claude-3-5-sonnet-20241022"
    temperature: float = 0.0
    max_tokens: int = 1024

    # API keys (loaded from environment)
    anthropic_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None

    def __post_init__(self):
        """Load API keys from environment."""
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")


@dataclass
class AppConfig:
    """Main application configuration."""

    pdf_processing: PDFProcessingConfig
    vector_store: VectorStoreConfig
    rag: RAGConfig

    @classmethod
    def default(cls) -> "AppConfig":
        """Create default configuration."""
        return cls(
            pdf_processing=PDFProcessingConfig(),
            vector_store=VectorStoreConfig(),
            rag=RAGConfig(),
        )

"""
Base PDF processor with shared logic for all embedding strategies.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import gc

from pypdf import PdfReader
import chromadb
from chromadb.config import Settings


class BasePDFProcessor(ABC):
    """
    Abstract base class for PDF processors.

    Handles all common logic:
    - ChromaDB initialization
    - PDF folder processing
    - Text extraction
    - Chunking algorithm
    - Batch management
    - Query interface

    Subclasses only need to implement:
    - _setup_collection(): Create ChromaDB collection with appropriate embedding function
    - _add_batch(): Add batch of texts with embeddings
    """

    def __init__(
        self,
        persist_directory: str,
        collection_name: str,
        chunk_size: int = 2000,
        chunk_overlap: int = 200,
        batch_size: int = 20,
    ):
        """
        Initialize processor.

        Args:
            persist_directory: Directory to persist ChromaDB data
            collection_name: Name of the collection
            chunk_size: Maximum characters per chunk
            chunk_overlap: Characters to overlap between chunks
            batch_size: Number of chunks to batch before adding to DB
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.batch_size = batch_size

        # Initialize ChromaDB client
        print("Initializing ChromaDB...")
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
            ),
        )

        # Let subclass set up collection with appropriate embedding function
        self.collection = self._setup_collection()

    @abstractmethod
    def _setup_collection(self):
        """
        Set up ChromaDB collection with appropriate embedding function.

        Returns:
            ChromaDB collection
        """
        pass

    @abstractmethod
    def _add_batch(self, texts: List[str], metadatas: List[Dict], ids: List[str]):
        """
        Add batch of documents with embeddings to collection.

        Args:
            texts: List of document texts
            metadatas: List of metadata dicts
            ids: List of unique IDs
        """
        pass

    def process_folder(self, folder_path: str) -> int:
        """
        Process all PDFs in folder.

        Args:
            folder_path: Path to folder containing PDFs

        Returns:
            Total number of chunks added
        """
        pdf_folder = Path(folder_path)
        if not pdf_folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        pdf_files = sorted(pdf_folder.glob("*.pdf"))
        if not pdf_files:
            raise FileNotFoundError(f"No PDFs found in {folder_path}")

        print(f"\nFound {len(pdf_files)} PDFs to process")
        total_chunks = 0

        for file_num, pdf_path in enumerate(pdf_files, start=1):
            print(f"\n[{file_num}/{len(pdf_files)}] Processing {pdf_path.name}...")

            try:
                chunks_added = self._process_single_pdf(pdf_path)
                total_chunks += chunks_added
                print(f"  ✓ {chunks_added} total chunks")

                # Force GC after each file
                gc.collect()

            except Exception as e:
                print(f"  ✗ Error: {e}")
                continue

        return total_chunks

    def _process_single_pdf(self, pdf_path: Path) -> int:
        """
        Process one PDF with batched additions to ChromaDB.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Number of chunks added
        """
        reader = PdfReader(pdf_path)
        total_pages = len(reader.pages)
        chunks_added = 0

        # Batch buffer
        batch_texts = []
        batch_metas = []
        batch_ids = []

        print(f"  Processing {total_pages} pages: ", end="", flush=True)

        for page_num, page in enumerate(reader.pages, start=1):
            # Show page progress
            if page_num % 10 == 0 or page_num == total_pages:
                print(f"[p{page_num}]", end="", flush=True)

            try:
                text = page.extract_text()
                if not text or not text.strip():
                    continue

                # Collect chunks for batching
                for chunk_idx, chunk_text in enumerate(self._chunk_text(text)):
                    if not chunk_text.strip():
                        continue

                    batch_texts.append(chunk_text)
                    batch_metas.append({
                        "source_file": pdf_path.name,
                        "page_number": page_num,
                        "total_pages": total_pages,
                        "chunk_index": chunk_idx,
                    })
                    batch_ids.append(f"{pdf_path.name}_p{page_num}_{chunk_idx}")

                    # Flush batch when it reaches batch_size
                    if len(batch_texts) >= self.batch_size:
                        self._add_batch(batch_texts, batch_metas, batch_ids)
                        chunks_added += len(batch_texts)
                        batch_texts = []
                        batch_metas = []
                        batch_ids = []

            except Exception as e:
                print(f"\n    Warning: Page {page_num} failed: {e}")
                continue

        # Flush remaining chunks
        if batch_texts:
            self._add_batch(batch_texts, batch_metas, batch_ids)
            chunks_added += len(batch_texts)

        print(f" ✓ {chunks_added} chunks")
        return chunks_added

    def _chunk_text(self, text: str):
        """
        Split text into overlapping chunks with sentence boundary detection.

        Yields:
            Text chunks
        """
        if len(text) <= self.chunk_size:
            yield text
            return

        start = 0
        chunks_created = 0
        max_chunks_per_page = 50  # Safety limit
        prev_start = -1

        while start < len(text) and start != prev_start:
            # Safety check
            if chunks_created >= max_chunks_per_page:
                print(
                    f"\n    WARNING: Hit chunk limit ({max_chunks_per_page}) for this page. "
                    f"Text length: {len(text)}. Skipping remainder.",
                    flush=True,
                )
                break

            end = min(start + self.chunk_size, len(text))
            chunk = text[start:end]

            # Try to break at sentence boundary
            if end < len(text):
                last_period = chunk.rfind(". ")
                last_newline = chunk.rfind("\n")
                break_point = max(last_period, last_newline)

                if break_point > self.chunk_size * 0.5:
                    chunk = chunk[: break_point + 1]
                    end = start + break_point + 1

            if chunk.strip():
                yield chunk
                chunks_created += 1

            # Move with overlap - track previous position to detect infinite loops
            prev_start = start
            start = end - self.chunk_overlap

            # If we're not making progress or past the end, stop
            if start <= prev_start or start >= len(text):
                break

    def query(self, query_text: str, top_k: int = 5):
        """
        Query the collection.

        Args:
            query_text: Query string
            top_k: Number of results

        Returns:
            Query results from ChromaDB
        """
        results = self.collection.query(
            query_texts=[query_text],
            n_results=top_k,
        )
        return results

    def count(self) -> int:
        """Get total chunks in collection."""
        return self.collection.count()

    def clear(self):
        """Clear the collection."""
        self.client.delete_collection(self.collection_name)
        self.collection = self._setup_collection()

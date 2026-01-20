"""
Ultra-lightweight streaming PDF processor that embeds one page at a time.
No batch accumulation - processes and stores immediately.
"""
from pathlib import Path
from typing import Optional
import gc

from pypdf import PdfReader
import chromadb
from chromadb.config import Settings


class UltraLightProcessor:
    """
    Memory-efficient processor that:
    1. Processes one page at a time
    2. Immediately embeds and stores (no accumulation)
    3. Uses ChromaDB's default embedding function (lighter than sentence-transformers)
    """

    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        collection_name: str = "pdf_documents",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        """
        Initialize processor.

        Args:
            persist_directory: Directory to persist ChromaDB data
            collection_name: Name of the collection
            chunk_size: Maximum characters per chunk
            chunk_overlap: Characters to overlap between chunks
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Initialize ChromaDB with default embedding function
        print("Initializing ChromaDB...")
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
            ),
        )

        # Get or create collection (uses default embedding function)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "PDF document chunks"},
        )
        print(f"✓ Collection ready: {self.collection.count()} existing chunks")

    def process_folder(self, folder_path: str) -> int:
        """
        Process all PDFs in folder with true streaming (no batching).

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
                print(f"  ✓ Added {chunks_added} chunks (total: {total_chunks})")

                # Force GC after each file
                gc.collect()

            except Exception as e:
                print(f"  ✗ Error: {e}")
                continue

        return total_chunks

    def _process_single_pdf(self, pdf_path: Path) -> int:
        """
        Process one PDF, storing chunks immediately (no accumulation).

        Args:
            pdf_path: Path to PDF file

        Returns:
            Number of chunks added
        """
        reader = PdfReader(pdf_path)
        total_pages = len(reader.pages)
        chunks_added = 0

        print(f"  Processing {total_pages} pages: ", end="", flush=True)

        for page_num, page in enumerate(reader.pages, start=1):
            # Show page progress
            if page_num % 5 == 0 or page_num == total_pages:
                print(f"[p{page_num}]", end="", flush=True)

            try:
                text = page.extract_text()
                if not text or not text.strip():
                    continue

                # Process and store chunks immediately
                page_chunks = self._store_page_chunks(
                    text=text,
                    source_file=pdf_path.name,
                    page_number=page_num,
                    total_pages=total_pages,
                )
                chunks_added += page_chunks

            except Exception as e:
                print(f"\n    Warning: Page {page_num} failed: {e}")
                continue

        print()  # New line after processing
        return chunks_added

    def _store_page_chunks(
        self,
        text: str,
        source_file: str,
        page_number: int,
        total_pages: int,
    ) -> int:
        """
        Chunk text and immediately store in ChromaDB (no accumulation).

        Args:
            text: Page text
            source_file: Source PDF filename
            page_number: Page number
            total_pages: Total pages in PDF

        Returns:
            Number of chunks stored
        """
        chunks_stored = 0

        # Generate chunks
        for chunk_idx, chunk_text in enumerate(self._chunk_text(text)):
            if not chunk_text.strip():
                continue

            # Create unique ID
            chunk_id = f"{source_file}_p{page_number}_{chunk_idx}"

            # Metadata
            metadata = {
                "source_file": source_file,
                "page_number": page_number,
                "total_pages": total_pages,
                "chunk_index": chunk_idx,
            }

            # Store immediately (ChromaDB handles embedding internally)
            try:
                self.collection.add(
                    documents=[chunk_text],
                    metadatas=[metadata],
                    ids=[chunk_id],
                )
                chunks_stored += 1

                # Show progress for every 5th chunk
                if chunks_stored % 5 == 0:
                    print(".", end="", flush=True)

            except Exception as e:
                print(f"\n      Warning: Failed to store chunk: {e}")
                continue

        if chunks_stored > 0:
            print(f" {chunks_stored}", end="", flush=True)

        return chunks_stored

    def _chunk_text(self, text: str):
        """
        Split text into overlapping chunks.

        Yields:
            Text chunks
        """
        if len(text) <= self.chunk_size:
            yield text
            return

        start = 0
        while start < len(text):
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

            # Move with overlap
            start = end - self.chunk_overlap
            if start < 0 or start >= len(text):
                break

    def query(self, query_text: str, top_k: int = 5):
        """
        Query the collection.

        Args:
            query_text: Query string
            top_k: Number of results

        Returns:
            Query results
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
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"description": "PDF document chunks"},
        )

"""
Fast PDF processing using pypdf (alternative to pdfplumber).
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List
import gc

from pypdf import PdfReader

from .pdf_processor import DocumentChunk


class FastPDFProcessor:
    """Faster PDF processor using pypdf instead of pdfplumber."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        batch_size: int = 3,
    ):
        """
        Initialize fast PDF processor.

        Args:
            chunk_size: Maximum characters per chunk
            chunk_overlap: Characters to overlap between chunks
            batch_size: Number of PDFs to process at once
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.batch_size = batch_size

    def process_folder(
        self, folder_path: str, file_pattern: str = "*.pdf"
    ) -> Iterator[List[DocumentChunk]]:
        """
        Process all PDFs in a folder in batches.

        Args:
            folder_path: Path to folder containing PDFs
            file_pattern: Glob pattern for PDF files

        Yields:
            Lists of DocumentChunk objects (one batch at a time)
        """
        pdf_folder = Path(folder_path)
        if not pdf_folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        pdf_files = sorted(pdf_folder.glob(file_pattern))
        if not pdf_files:
            raise FileNotFoundError(f"No PDFs found in {folder_path}")

        print(f"Found {len(pdf_files)} PDFs to process")

        # Process in batches
        for i in range(0, len(pdf_files), self.batch_size):
            batch_files = pdf_files[i : i + self.batch_size]
            batch_chunks = []

            print(
                f"\nProcessing batch {i // self.batch_size + 1}: "
                f"{len(batch_files)} files"
            )

            for pdf_path in batch_files:
                try:
                    print(f"  Processing {pdf_path.name}...", end=" ", flush=True)
                    chunks = list(self._process_single_pdf(pdf_path))
                    batch_chunks.extend(chunks)
                    print(f"✓ {len(chunks)} chunks")
                except Exception as e:
                    print(f"✗ Error: {str(e)}")
                    continue

            yield batch_chunks

            # Force garbage collection after each batch
            del batch_chunks
            gc.collect()

    def _process_single_pdf(self, pdf_path: Path) -> Iterator[DocumentChunk]:
        """
        Process a single PDF file using pypdf (faster).

        Args:
            pdf_path: Path to PDF file

        Yields:
            DocumentChunk objects
        """
        reader = PdfReader(pdf_path)
        total_pages = len(reader.pages)

        for page_num, page in enumerate(reader.pages, start=1):
            try:
                # Extract text (much faster than pdfplumber)
                text = page.extract_text()
                if not text or not text.strip():
                    continue

                # Create chunks from page text
                for chunk_text in self._chunk_text(text):
                    metadata = {
                        "source_file": pdf_path.name,
                        "page_number": page_num,
                        "total_pages": total_pages,
                        "file_path": str(pdf_path),
                    }

                    yield DocumentChunk(text=chunk_text, metadata=metadata)

            except Exception as e:
                # Skip problematic pages
                print(f"\n      Warning: Page {page_num} failed: {e}")
                continue

    def _chunk_text(self, text: str) -> Iterator[str]:
        """
        Split text into overlapping chunks.

        Args:
            text: Text to chunk

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

            # Try to break at sentence or word boundary
            if end < len(text):
                # Look for sentence end
                last_period = chunk.rfind(". ")
                last_newline = chunk.rfind("\n")
                break_point = max(last_period, last_newline)

                if break_point > self.chunk_size * 0.5:  # At least 50% of chunk
                    chunk = chunk[: break_point + 1]
                    end = start + break_point + 1

            if chunk.strip():
                yield chunk

            # Move start with overlap
            start = end - self.chunk_overlap
            if start < 0:
                start = 0

            # Prevent infinite loop
            if start >= len(text):
                break

"""
PDF processor using sentence-transformers embeddings with ChromaDB.
"""
from pathlib import Path
from typing import Iterator, List
import gc

from pypdf import PdfReader
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


class SentenceTransformerProcessor:
    """
    RAG processor using sentence-transformers for embeddings.
    More memory-intensive but better semantic search than ONNX.
    """

    def __init__(
        self,
        persist_directory: str = "./chroma_db_sentence_transformers",
        collection_name: str = "pdf_sentence_transformers",
        model_name: str = "all-MiniLM-L6-v2",
        chunk_size: int = 2000,
        chunk_overlap: int = 200,
    ):
        """
        Initialize processor.

        Args:
            persist_directory: ChromaDB persistence directory
            collection_name: Collection name
            model_name: SentenceTransformer model
            chunk_size: Characters per chunk
            chunk_overlap: Overlap between chunks
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Initialize ChromaDB
        print(f"Initializing ChromaDB with sentence-transformers...")
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )

        # Load embedding model
        print(f"Loading {model_name}...")
        self.embedding_model = SentenceTransformer(model_name)
        print(f"✓ Model loaded (dim={self.embedding_model.get_sentence_embedding_dimension()})")

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"embedding_model": model_name},
        )

    def process_folder(self, folder_path: str) -> int:
        """Process all PDFs in folder."""
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
                reader = PdfReader(pdf_path)
                total_pages = len(reader.pages)
                print(f"  Processing {total_pages} pages: ", end="", flush=True)

                batch_texts = []
                batch_metas = []
                batch_ids = []

                for page_num, page in enumerate(reader.pages, start=1):
                    if page_num % 10 == 0 or page_num == total_pages:
                        print(f"[p{page_num}]", end="", flush=True)

                    text = page.extract_text()
                    if not text or not text.strip():
                        continue

                    # Chunk the page text
                    for chunk_idx, chunk_text in enumerate(self._chunk_text(text)):
                        batch_texts.append(chunk_text)
                        batch_metas.append({
                            "source_file": pdf_path.name,
                            "page_number": page_num,
                            "total_pages": total_pages,
                            "chunk_index": chunk_idx,
                        })
                        batch_ids.append(f"{pdf_path.name}_p{page_num}_{chunk_idx}")

                        # Flush batch every 25 chunks
                        if len(batch_texts) >= 25:
                            self._add_batch(batch_texts, batch_metas, batch_ids)
                            print(".", end="", flush=True)
                            total_chunks += len(batch_texts)
                            batch_texts = []
                            batch_metas = []
                            batch_ids = []

                # Flush remaining
                if batch_texts:
                    self._add_batch(batch_texts, batch_metas, batch_ids)
                    total_chunks += len(batch_texts)

                print(f" ✓ {total_chunks} total chunks")

                gc.collect()

            except Exception as e:
                print(f"  ✗ Error: {e}")
                continue

        return total_chunks

    def _add_batch(self, texts: List[str], metadatas: List[dict], ids: List[str]):
        """Add a batch of chunks with embeddings."""
        # Generate embeddings
        embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).tolist()

        # Add to collection
        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )

        del embeddings
        gc.collect()

    def _chunk_text(self, text: str) -> Iterator[str]:
        """Split text into overlapping chunks."""
        if len(text) <= self.chunk_size:
            yield text
            return

        start = 0
        prev_start = -1
        chunks_created = 0
        max_chunks = 50

        while start < len(text) and start != prev_start:
            if chunks_created >= max_chunks:
                break

            end = min(start + self.chunk_size, len(text))
            chunk = text[start:end]

            # Break at sentence boundary
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

            prev_start = start
            start = end - self.chunk_overlap

            if start <= prev_start or start >= len(text):
                break

    def query(self, query_text: str, top_k: int = 5):
        """Query the collection."""
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
        )

        return results

    def count(self) -> int:
        """Get total chunks."""
        return self.collection.count()

    def clear(self):
        """Clear the collection."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"embedding_model": self.model_name},
        )

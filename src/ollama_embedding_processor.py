"""
PDF processor using Ollama embeddings (nomic-embed-text).
"""
from pathlib import Path
from typing import Iterator, List
import gc

from pypdf import PdfReader
import chromadb
from chromadb.config import Settings
import ollama


class OllamaEmbeddingProcessor:
    """
    RAG processor using Ollama's nomic-embed-text model.
    Different architecture from all-MiniLM-L6-v2, optimized for RAG.
    """

    def __init__(
        self,
        persist_directory: str = "./chroma_db_ollama",
        collection_name: str = "pdf_ollama_embeddings",
        model_name: str = "nomic-embed-text",
        chunk_size: int = 2000,
        chunk_overlap: int = 200,
    ):
        """
        Initialize processor with Ollama embeddings.

        Args:
            persist_directory: ChromaDB persistence directory
            collection_name: Collection name
            model_name: Ollama embedding model (e.g., nomic-embed-text)
            chunk_size: Characters per chunk
            chunk_overlap: Overlap between chunks
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

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

        # Initialize ChromaDB
        print(f"Initializing ChromaDB...")
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"embedding_model": model_name, "dimension": self.embedding_dim},
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

                        # Flush batch every 20 chunks (Ollama can be slower)
                        if len(batch_texts) >= 20:
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
        """Add a batch of chunks with Ollama embeddings."""
        # Generate embeddings via Ollama
        embeddings = []
        for text in texts:
            response = ollama.embeddings(model=self.model_name, prompt=text)
            embeddings.append(response['embedding'])

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
        # Generate query embedding via Ollama
        response = ollama.embeddings(model=self.model_name, prompt=query_text)
        query_embedding = response['embedding']

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
            metadata={"embedding_model": self.model_name, "dimension": self.embedding_dim},
        )

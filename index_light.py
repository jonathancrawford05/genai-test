#!/usr/bin/env python3
"""
Ultra-lightweight indexing script using ChromaDB's default embeddings.
This avoids loading sentence-transformers and PyTorch entirely.
"""
import sys
import time
from dotenv import load_dotenv

from src.ultra_light_processor import UltraLightProcessor

def main():
    load_dotenv()

    print("=" * 60)
    print("ULTRA-LIGHT PDF INDEXING")
    print("(Using ChromaDB's default embedding function)")
    print("=" * 60)

    # Initialize processor (larger chunks, batched additions)
    processor = UltraLightProcessor(
        persist_directory="./chroma_db_light",
        collection_name="pdf_documents_light",
        chunk_size=2000,  # Larger chunks = fewer total chunks
        chunk_overlap=200,
        batch_size=20,  # Add 20 chunks at a time (much faster)
    )

    # Check if already indexed
    existing = processor.count()
    if existing > 0:
        print(f"\n⚠️  Database already contains {existing} chunks")
        response = input("Clear and re-index? (y/n): ")
        if response.lower() == "y":
            print("Clearing...")
            processor.clear()
        else:
            print("Using existing index")
            return

    # Process PDFs
    folder = sys.argv[1] if len(sys.argv) > 1 else "artifacts/1"
    print(f"\nIndexing folder: {folder}")

    start_time = time.time()
    total_chunks = processor.process_folder(folder)
    elapsed = time.time() - start_time

    print("\n" + "=" * 60)
    print("✓ INDEXING COMPLETE")
    print(f"  Total chunks: {total_chunks}")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Speed: {total_chunks / elapsed:.1f} chunks/sec")
    print("=" * 60)

    # Test query
    print("\n\nTesting query...")
    results = processor.query("What are the rating rules?", top_k=3)

    print("\nTop 3 results:")
    for i, doc in enumerate(results["documents"][0], 1):
        meta = results["metadatas"][0][i - 1]
        print(f"\n{i}. {meta['source_file']} (p{meta['page_number']})")
        print(f"   {doc[:150]}...")

if __name__ == "__main__":
    main()

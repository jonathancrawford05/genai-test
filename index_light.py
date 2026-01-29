#!/usr/bin/env python3
"""
Lightweight PDF indexing script using ONNX embeddings.
Memory-efficient with fast ONNX runtime.
"""
import sys
import time
from dotenv import load_dotenv

from src.onnx_processor import ONNXProcessor

def main():
    load_dotenv()

    print("=" * 60)
    print("PDF INDEXING - ONNX EMBEDDINGS")
    print("(all-MiniLM-L6-v2, 79MB, 384 dims)")
    print("=" * 60)

    # Initialize processor
    processor = ONNXProcessor(
        persist_directory="./chroma_db_onnx",
        collection_name="pdf_documents",
        chunk_size=2000,
        chunk_overlap=200,
        batch_size=20,
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

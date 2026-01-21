#!/usr/bin/env python3
"""
Query script for the ultra-light RAG system.
"""
import sys
from src.ultra_light_processor import UltraLightProcessor

def main():
    # Initialize processor (connected to existing database)
    processor = UltraLightProcessor(
        persist_directory="./chroma_db_light",
        collection_name="pdf_documents_light",
    )

    print(f"\n{'=' * 60}")
    print(f"RAG QUERY SYSTEM")
    print(f"Database: {processor.count()} chunks indexed")
    print(f"{'=' * 60}\n")

    # Get question
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = input("Enter your question: ")

    print(f"\nQuestion: {question}\n")

    # Query
    results = processor.query(question, top_k=5)

    # Display results
    print("Top 5 relevant chunks:\n")
    for i, doc in enumerate(results["documents"][0], 1):
        meta = results["metadatas"][0][i - 1]
        distance = results["distances"][0][i - 1]

        print(f"{'-' * 60}")
        print(f"Result {i} - {meta['source_file']} (page {meta['page_number']})")
        print(f"Relevance score: {1 - distance:.3f}")
        print(f"{'-' * 60}")
        print(f"{doc[:500]}...")
        print()

    print("\n" + "=" * 60)
    print("Note: This uses keyword-based retrieval (TF-IDF).")
    print("For semantic search with LLM answers, use main.py with")
    print("the full RAG system (requires API key).")
    print("=" * 60)

if __name__ == "__main__":
    main()

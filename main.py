#!/usr/bin/env python3
"""
Main application for RAG-based PDF question answering.

Usage:
    # Index PDFs
    python main.py index artifacts/1

    # Answer questions from CSV
    python main.py query artifacts/questions.csv

    # Answer a single question
    python main.py ask "What are the rating plan rules?" artifacts/1

    # Clear the database
    python main.py clear
"""
import argparse
import csv
import sys
import time
from pathlib import Path
from typing import List

from dotenv import load_dotenv

from src.config import AppConfig
from src.pdf_processor import PDFProcessor
from src.fast_pdf_processor import FastPDFProcessor
from src.vector_store import VectorStore
from src.rag_engine import RAGEngine


def index_pdfs(pdf_folder: str, config: AppConfig) -> VectorStore:
    """
    Index PDFs from a folder into the vector store.

    Args:
        pdf_folder: Path to folder containing PDFs
        config: Application configuration

    Returns:
        Initialized VectorStore
    """
    print(f"\n{'=' * 60}")
    print("INDEXING PDFs")
    print(f"{'=' * 60}\n")

    # Initialize components (use FastPDFProcessor for better performance)
    processor = FastPDFProcessor(
        chunk_size=config.pdf_processing.chunk_size,
        chunk_overlap=config.pdf_processing.chunk_overlap,
        batch_size=config.pdf_processing.batch_size,
    )

    vector_store = VectorStore(
        persist_directory=config.vector_store.persist_directory,
        collection_name=config.vector_store.collection_name,
        embedding_model=config.vector_store.embedding_model,
    )

    # Check if already indexed
    existing_count = vector_store.count()
    if existing_count > 0:
        print(f"⚠️  Vector store already contains {existing_count} chunks")
        response = input("Clear and re-index? (y/n): ")
        if response.lower() == "y":
            print("Clearing existing data...")
            vector_store.clear()
        else:
            print("Using existing index")
            return vector_store

    # Process PDFs in batches
    start_time = time.time()
    total_chunks = 0

    for batch_chunks in processor.process_folder(pdf_folder):
        num_added = vector_store.add_chunks(batch_chunks)
        total_chunks += num_added
        print(f"  → Added {num_added} chunks to vector store")

    elapsed = time.time() - start_time

    print(f"\n{'=' * 60}")
    print(f"✓ Indexing complete!")
    print(f"  Total chunks: {total_chunks}")
    print(f"  Time: {elapsed:.2f}s")
    print(f"{'=' * 60}\n")

    return vector_store


def query_from_csv(csv_path: str, config: AppConfig) -> List[dict]:
    """
    Answer questions from a CSV file.

    Args:
        csv_path: Path to CSV file with questions
        config: Application configuration

    Returns:
        List of results
    """
    print(f"\n{'=' * 60}")
    print("ANSWERING QUESTIONS FROM CSV")
    print(f"{'=' * 60}\n")

    # Load vector store
    vector_store = VectorStore(
        persist_directory=config.vector_store.persist_directory,
        collection_name=config.vector_store.collection_name,
        embedding_model=config.vector_store.embedding_model,
    )

    if vector_store.count() == 0:
        print("❌ Vector store is empty! Please run 'index' first.")
        sys.exit(1)

    print(f"Vector store loaded: {vector_store.count()} chunks\n")

    # Initialize RAG engine
    rag_engine = RAGEngine(
        vector_store=vector_store,
        model_provider=config.rag.model_provider,
        model_name=config.rag.model_name,
        temperature=config.rag.temperature,
        max_tokens=config.rag.max_tokens,
    )

    # Load questions
    questions_data = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            questions_data.append(row)

    print(f"Loaded {len(questions_data)} questions from {csv_path}\n")

    # Answer questions
    results = []
    for i, row in enumerate(questions_data, 1):
        question_id = row["id"]
        question = row["question"]
        expected = row.get("expected_output", "")

        print(f"\n{'-' * 60}")
        print(f"Question {i}/{len(questions_data)} (ID: {question_id})")
        print(f"Q: {question}")
        print(f"-" * 60)

        result = rag_engine.answer_question(
            question=question,
            top_k=config.vector_store.top_k,
        )

        print(f"\nA: {result['answer']}\n")

        if expected:
            print(f"Expected: {expected[:200]}...")

        if result.get("sources"):
            print(f"\nSources:")
            for src in result["sources"][:3]:
                print(
                    f"  - {src['source_file']} (p{src['page_number']}) "
                    f"[score: {src['relevance_score']:.3f}]"
                )

        results.append(
            {
                "id": question_id,
                "question": question,
                "answer": result["answer"],
                "expected": expected,
                "num_sources": result["num_sources"],
            }
        )

    print(f"\n{'=' * 60}")
    print(f"✓ Answered {len(results)} questions")
    print(f"{'=' * 60}\n")

    return results


def ask_question(question: str, pdf_folder: str, config: AppConfig) -> dict:
    """
    Answer a single question.

    Args:
        question: Question to answer
        pdf_folder: PDF folder (for context)
        config: Application configuration

    Returns:
        Result dictionary
    """
    # Load vector store
    vector_store = VectorStore(
        persist_directory=config.vector_store.persist_directory,
        collection_name=config.vector_store.collection_name,
        embedding_model=config.vector_store.embedding_model,
    )

    if vector_store.count() == 0:
        print("❌ Vector store is empty! Please run 'index' first.")
        sys.exit(1)

    # Initialize RAG engine
    rag_engine = RAGEngine(
        vector_store=vector_store,
        model_provider=config.rag.model_provider,
        model_name=config.rag.model_name,
        temperature=config.rag.temperature,
        max_tokens=config.rag.max_tokens,
    )

    # Answer question
    print(f"\nQuestion: {question}\n")
    result = rag_engine.answer_question(
        question=question,
        top_k=config.vector_store.top_k,
    )

    print(f"Answer: {result['answer']}\n")

    if result.get("sources"):
        print(f"Sources:")
        for src in result["sources"]:
            print(
                f"  - {src['source_file']} (p{src['page_number']}) "
                f"[score: {src['relevance_score']:.3f}]"
            )

    return result


def clear_database(config: AppConfig):
    """Clear the vector database."""
    vector_store = VectorStore(
        persist_directory=config.vector_store.persist_directory,
        collection_name=config.vector_store.collection_name,
        embedding_model=config.vector_store.embedding_model,
    )

    count = vector_store.count()
    if count > 0:
        print(f"Clearing {count} chunks from vector store...")
        vector_store.clear()
        print("✓ Database cleared")
    else:
        print("Database is already empty")


def main():
    """Main entry point."""
    # Load environment variables
    load_dotenv()

    # Parse arguments
    parser = argparse.ArgumentParser(
        description="RAG-based PDF question answering system"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Index command
    index_parser = subparsers.add_parser("index", help="Index PDFs")
    index_parser.add_argument("pdf_folder", help="Path to PDF folder")

    # Query command
    query_parser = subparsers.add_parser("query", help="Answer questions from CSV")
    query_parser.add_argument("csv_file", help="Path to CSV file")

    # Ask command
    ask_parser = subparsers.add_parser("ask", help="Ask a single question")
    ask_parser.add_argument("question", help="Question to ask")
    ask_parser.add_argument("pdf_folder", help="PDF folder (for context)")

    # Clear command
    subparsers.add_parser("clear", help="Clear the vector database")

    # Stats command
    subparsers.add_parser("stats", help="Show database statistics")

    args = parser.parse_args()

    # Load configuration
    config = AppConfig.default()

    # Execute command
    if args.command == "index":
        index_pdfs(args.pdf_folder, config)

    elif args.command == "query":
        query_from_csv(args.csv_file, config)

    elif args.command == "ask":
        ask_question(args.question, args.pdf_folder, config)

    elif args.command == "clear":
        clear_database(config)

    elif args.command == "stats":
        vector_store = VectorStore(
            persist_directory=config.vector_store.persist_directory,
            collection_name=config.vector_store.collection_name,
            embedding_model=config.vector_store.embedding_model,
        )
        stats = vector_store.get_stats()
        print("\nVector Store Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Generate document summaries for all PDFs in the artifacts folder.

This creates a JSON file with structured summaries that can be manually
reviewed and edited before being used by the Router agent.

Usage:
    python generate_summaries.py [--folder artifacts/1] [--model llama3.2]
"""
import argparse
from pathlib import Path

from src.agents.document_summarizer import DocumentSummarizer


def main():
    parser = argparse.ArgumentParser(description="Generate PDF document summaries")
    parser.add_argument(
        '--folder',
        default='artifacts/1',
        help='Folder containing PDFs (default: artifacts/1)'
    )
    parser.add_argument(
        '--output',
        default='artifacts/document_summaries.json',
        help='Output JSON file (default: artifacts/document_summaries.json)'
    )
    parser.add_argument(
        '--model',
        default='llama3.2',
        help='Ollama model for summarization (default: llama3.2)'
    )
    parser.add_argument(
        '--sample-pages',
        type=int,
        default=3,
        help='Number of random pages to sample per PDF (default: 3)'
    )

    args = parser.parse_args()

    # Initialize summarizer
    summarizer = DocumentSummarizer(model=args.model)

    # Set sample pages (will be used by summarizer)
    summarizer.sample_pages = args.sample_pages

    # Generate summaries
    folder_path = Path(args.folder)
    output_path = Path(args.output)

    if not folder_path.exists():
        print(f"Error: Folder not found: {folder_path}")
        return 1

    summaries = summarizer.summarize_folder(folder_path, output_path)

    # Print summary statistics
    print(f"\n{'=' * 60}")
    print("SUMMARY STATISTICS")
    print(f"{'=' * 60}")
    print(f"Total documents: {len(summaries)}")

    # Group by document type
    by_type = {}
    for filename, data in summaries.items():
        doc_type = data.get('document_type', 'unknown')
        by_type[doc_type] = by_type.get(doc_type, 0) + 1

    print("\nBy document type:")
    for doc_type, count in sorted(by_type.items()):
        print(f"  {doc_type}: {count}")

    # Check for errors
    errors = [f for f, d in summaries.items() if 'error' in d]
    if errors:
        print(f"\n⚠️  {len(errors)} documents had errors:")
        for f in errors:
            print(f"  - {f}")

    print(f"\n{'=' * 60}")
    print("NEXT STEPS")
    print(f"{'=' * 60}")
    print(f"1. Review summaries: {output_path}")
    print(f"2. Edit/refine as needed (it's a JSON file)")
    print(f"3. Summaries will be used by Router agent for document selection")
    print()

    return 0


if __name__ == "__main__":
    exit(main())

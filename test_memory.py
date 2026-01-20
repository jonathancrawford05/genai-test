#!/usr/bin/env python3
"""
Test script to identify memory bottleneck.
"""
import sys
from pathlib import Path
from pypdf import PdfReader

def test_pdf_reading(pdf_path: str):
    """Test just reading a PDF without any processing."""
    print(f"Testing PDF reading: {pdf_path}")

    reader = PdfReader(pdf_path)
    print(f"  Total pages: {len(reader.pages)}")

    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        print(f"  Page {page_num}: {len(text)} chars")
        if page_num >= 3:  # Just test first 3 pages
            break

    print("✓ PDF reading works fine")

def test_chunking(pdf_path: str):
    """Test chunking without embeddings."""
    print(f"\nTesting chunking: {pdf_path}")

    reader = PdfReader(pdf_path)
    chunk_count = 0

    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if not text:
            continue

        # Simple chunking
        chunk_size = 1000
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i + chunk_size]
            chunk_count += 1

        print(f"  Page {page_num}: {chunk_count} total chunks so far")
        if page_num >= 3:
            break

    print(f"✓ Chunking works fine: {chunk_count} chunks")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_memory.py <pdf_path>")
        sys.exit(1)

    pdf_path = sys.argv[1]

    try:
        test_pdf_reading(pdf_path)
        test_chunking(pdf_path)
        print("\n✓ All tests passed!")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()

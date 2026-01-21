#!/usr/bin/env python3
"""
Debug script to see what's actually in the PDF.
"""
import sys
from pypdf import PdfReader

def analyze_pdf(pdf_path: str):
    """Analyze PDF content."""
    print(f"Analyzing: {pdf_path}\n")

    reader = PdfReader(pdf_path)
    print(f"Total pages: {len(reader.pages)}\n")

    for page_num, page in enumerate(reader.pages[:3], start=1):  # First 3 pages
        text = page.extract_text()

        print(f"{'=' * 60}")
        print(f"PAGE {page_num}")
        print(f"{'=' * 60}")
        print(f"Character count: {len(text)}")
        print(f"Line count: {len(text.splitlines())}")
        print(f"\nFirst 500 characters:")
        print(text[:500])
        print(f"\n... (showing first 500 of {len(text)} chars)")

        # Show what chunks would be created
        chunk_size = 2000
        overlap = 200
        num_chunks = 0

        if len(text) <= chunk_size:
            num_chunks = 1
        else:
            start = 0
            prev_start = -1
            while start < len(text) and start != prev_start:
                num_chunks += 1
                end = min(start + chunk_size, len(text))
                prev_start = start
                start = end - overlap
                if start >= len(text):
                    break

        print(f"\nChunks this page would create: {num_chunks}")
        print()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        pdf_path = "artifacts/1/(129371443-179747531)-Miccolis ILF paper.pdf"
    else:
        pdf_path = sys.argv[1]

    analyze_pdf(pdf_path)

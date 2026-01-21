#!/usr/bin/env python3
"""
Unified RAG query script supporting multiple LLM providers.

Providers:
  - ollama (local, free)
  - openai (API key required)
  - anthropic (API key required)

Usage:
  python query_rag.py "your question"
  python query_rag.py --provider ollama --model llama3.2 "your question"
  python query_rag.py --provider openai --model gpt-4o-mini "your question"
  python query_rag.py --provider anthropic --model claude-3-5-sonnet-20241022 "your question"
"""
import sys
import os
import argparse
from dotenv import load_dotenv

from src.ultra_light_processor import UltraLightProcessor


def answer_with_ollama(question: str, processor: UltraLightProcessor, model: str, top_k: int):
    """Answer using Ollama."""
    import ollama

    results = processor.query(question, top_k=top_k)
    if not results["documents"][0]:
        return "No relevant information found."

    context = build_context(results)
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{context}\n\nQuestion: {question}\n\nAnswer:"},
        ],
        options={"temperature": 0, "num_predict": 1024}
    )
    return response["message"]["content"].strip(), extract_sources(results)


def answer_with_openai(question: str, processor: UltraLightProcessor, model: str, top_k: int):
    """Answer using OpenAI."""
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    results = processor.query(question, top_k=top_k)
    if not results["documents"][0]:
        return "No relevant information found."

    context = build_context(results)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{context}\n\nQuestion: {question}\n\nAnswer:"},
        ],
        temperature=0,
        max_tokens=1024,
    )
    return response.choices[0].message.content.strip(), extract_sources(results)


def answer_with_anthropic(question: str, processor: UltraLightProcessor, model: str, top_k: int):
    """Answer using Anthropic."""
    from anthropic import Anthropic

    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    results = processor.query(question, top_k=top_k)
    if not results["documents"][0]:
        return "No relevant information found."

    context = build_context(results)
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        temperature=0,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": f"{context}\n\nQuestion: {question}\n\nAnswer:"}
        ],
    )
    return response.content[0].text.strip(), extract_sources(results)


def build_context(results):
    """Build context from query results."""
    context_parts = []
    for i, doc in enumerate(results["documents"][0], 1):
        meta = results["metadatas"][0][i - 1]
        context_parts.append(
            f"[Source {i}] {meta['source_file']} (Page {meta['page_number']}):\n{doc}"
        )
    return "Context from relevant documents:\n\n" + "\n\n".join(context_parts)


def extract_sources(results):
    """Extract sources from query results."""
    sources = []
    for i, doc in enumerate(results["documents"][0], 1):
        meta = results["metadatas"][0][i - 1]
        distance = results["distances"][0][i - 1]
        sources.append({
            "source_file": meta["source_file"],
            "page_number": meta["page_number"],
            "relevance_score": 1 - distance,
        })
    return sources


SYSTEM_PROMPT = """You are a helpful AI assistant that answers questions based on provided context from PDF documents.

Guidelines:
- Answer the question using ONLY the information from the provided context
- Be precise and accurate
- If the context contains numerical data, extract it exactly as shown
- If the question asks for a list, format your response as a clear bulleted list with * prefix
- If the question asks for a calculation, provide just the final number with $ prefix if it's currency
- If the context doesn't contain enough information, say so
- Cite sources by mentioning the document name and page number when relevant
- Do not make up information or use outside knowledge"""


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="RAG query system with multiple LLM providers"
    )
    parser.add_argument("question", nargs="*", help="Question to ask")
    parser.add_argument(
        "--provider",
        choices=["ollama", "openai", "anthropic"],
        default="ollama",
        help="LLM provider (default: ollama)"
    )
    parser.add_argument(
        "--model",
        help="Model name (default: llama3.2 for Ollama, gpt-4o-mini for OpenAI, claude-3-5-sonnet for Anthropic)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve (default: 5)"
    )

    args = parser.parse_args()

    # Default models
    if not args.model:
        if args.provider == "ollama":
            args.model = "llama3.2"
        elif args.provider == "openai":
            args.model = "gpt-4o-mini"
        elif args.provider == "anthropic":
            args.model = "claude-3-5-sonnet-20241022"

    # Get question
    question = " ".join(args.question) if args.question else input("Enter your question: ")

    # Initialize processor
    processor = UltraLightProcessor(
        persist_directory="./chroma_db_light",
        collection_name="pdf_documents_light",
    )

    print(f"\n{'=' * 60}")
    print(f"RAG SYSTEM")
    print(f"Provider: {args.provider.upper()}")
    print(f"Model: {args.model}")
    print(f"Database: {processor.count()} chunks indexed")
    print(f"{'=' * 60}\n")

    print(f"Question: {question}\n")
    print("Retrieving relevant documents...")

    # Answer question
    try:
        if args.provider == "ollama":
            answer, sources = answer_with_ollama(question, processor, args.model, args.top_k)
        elif args.provider == "openai":
            answer, sources = answer_with_openai(question, processor, args.model, args.top_k)
        elif args.provider == "anthropic":
            answer, sources = answer_with_anthropic(question, processor, args.model, args.top_k)

        print(f"\n{'=' * 60}")
        print("ANSWER")
        print(f"{'=' * 60}\n")
        print(answer)

        print(f"\n{'=' * 60}")
        print("SOURCES")
        print(f"{'=' * 60}\n")

        for i, src in enumerate(sources, 1):
            print(f"{i}. {src['source_file']} (page {src['page_number']}) "
                  f"[relevance: {src['relevance_score']:.3f}]")

        print()

    except Exception as e:
        print(f"\n❌ Error: {e}")

        if args.provider == "ollama":
            print("\nOllama troubleshooting:")
            print("1. Is Ollama running? (ollama serve)")
            print(f"2. Is the model installed? (ollama pull {args.model})")
            print("3. List available models: ollama list")
        elif args.provider in ["openai", "anthropic"]:
            api_key_name = f"{args.provider.upper()}_API_KEY"
            print(f"\n{args.provider.title()} troubleshooting:")
            print(f"1. Is {api_key_name} set in .env file?")
            print("2. Do you have sufficient API credits?")

        sys.exit(1)


if __name__ == "__main__":
    main()

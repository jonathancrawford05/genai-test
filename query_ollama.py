#!/usr/bin/env python3
"""
RAG system with Ollama (local LLM) using the ultra-light database.
"""
import sys
import ollama

from src.onnx_processor import ONNXProcessor


def answer_question_with_ollama(
    question: str,
    processor: ONNXProcessor,
    model: str = "llama3.2",
    top_k: int = 5
) -> dict:
    """
    Answer a question using RAG with Ollama.

    Args:
        question: User question
        processor: Ultra-light processor with indexed data
        model: Ollama model name (e.g., "llama3.2", "mistral", "gemma2")
        top_k: Number of chunks to retrieve

    Returns:
        Dictionary with answer and sources
    """
    # Retrieve relevant chunks
    results = processor.query(question, top_k=top_k)

    if not results["documents"][0]:
        return {
            "answer": "I couldn't find any relevant information to answer this question.",
            "sources": [],
        }

    # Build context from retrieved chunks
    context_parts = []
    sources = []

    for i, doc in enumerate(results["documents"][0], 1):
        meta = results["metadatas"][0][i - 1]
        distance = results["distances"][0][i - 1]

        context_parts.append(
            f"[Source {i}] {meta['source_file']} (Page {meta['page_number']}):\n{doc}"
        )

        sources.append({
            "source_file": meta["source_file"],
            "page_number": meta["page_number"],
            "relevance_score": 1 - distance,
        })

    context = "\n\n".join(context_parts)

    # Create prompt
    system_prompt = """You are a helpful AI assistant that answers questions based on provided context from PDF documents.

Guidelines:
- Answer the question using ONLY the information from the provided context
- Be precise and accurate
- If the context contains numerical data, extract it exactly as shown
- If the question asks for a list, format your response as a clear bulleted list with * prefix
- If the question asks for a calculation, provide just the final number with $ prefix if it's currency
- If the context doesn't contain enough information, say so
- Cite sources by mentioning the document name and page number when relevant
- Do not make up information or use outside knowledge"""

    user_prompt = f"""Context from relevant documents:

{context}

Question: {question}

Answer:"""

    # Call Ollama
    print(f"Calling Ollama ({model})...", flush=True)

    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        options={
            "temperature": 0,
            "num_predict": 1024,
        }
    )

    answer = response["message"]["content"]

    return {
        "answer": answer.strip(),
        "sources": sources,
        "question": question,
        "model": model,
    }


def main():
    # Parse arguments
    model = "llama3.2"  # Default model
    question = None

    args = sys.argv[1:]
    if args:
        # Check for --model flag
        if "--model" in args:
            model_idx = args.index("--model")
            if model_idx + 1 < len(args):
                model = args[model_idx + 1]
                args = args[:model_idx] + args[model_idx + 2:]

        # Remaining args are the question
        if args:
            question = " ".join(args)

    # Initialize processor with existing database
    processor = ONNXProcessor(
        persist_directory="./chroma_db_light",
        collection_name="pdf_documents_light",
    )

    print(f"\n{'=' * 60}")
    print(f"RAG SYSTEM WITH OLLAMA")
    print(f"Database: {processor.count()} chunks indexed")
    print(f"Model: {model}")
    print(f"{'=' * 60}\n")

    # Get question if not provided
    if not question:
        question = input("Enter your question: ")

    print(f"\nQuestion: {question}\n")
    print("Retrieving relevant documents...")

    # Answer question
    try:
        result = answer_question_with_ollama(question, processor, model=model, top_k=5)

        print(f"\n{'=' * 60}")
        print("ANSWER")
        print(f"{'=' * 60}\n")
        print(result["answer"])

        print(f"\n{'=' * 60}")
        print("SOURCES")
        print(f"{'=' * 60}\n")

        for i, src in enumerate(result["sources"], 1):
            print(f"{i}. {src['source_file']} (page {src['page_number']}) "
                  f"[relevance: {src['relevance_score']:.3f}]")

        print()

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Is Ollama running? (ollama serve)")
        print(f"2. Is the model installed? (ollama pull {model})")
        print("3. Available models: ollama list")
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
RAG system with OpenAI using the ultra-light database.
Combines keyword retrieval with LLM-powered answers.
"""
import sys
import os
from dotenv import load_dotenv
from openai import OpenAI

from src.onnx_processor import ONNXProcessor


def answer_question_with_llm(question: str, processor: ONNXProcessor, top_k: int = 5) -> dict:
    """
    Answer a question using RAG with OpenAI.

    Args:
        question: User question
        processor: Ultra-light processor with indexed data
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
- If the question asks for a list, format your response as a clear bulleted list
- If the question asks for a calculation, show your work
- If the context doesn't contain enough information, say so
- Cite sources by mentioning the document name and page number when relevant
- Do not make up information or use outside knowledge"""

    user_prompt = f"""Context from relevant documents:

{context}

Question: {question}

Answer:"""

    # Call OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.chat.completions.create(
        model="gpt-4o-mini",  # Fast and cheap
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        max_tokens=1024,
    )

    answer = response.choices[0].message.content

    return {
        "answer": answer.strip(),
        "sources": sources,
        "question": question,
    }


def main():
    load_dotenv()

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not found in environment")
        print("Please add it to your .env file:")
        print("OPENAI_API_KEY=your_key_here")
        sys.exit(1)

    # Initialize processor with existing database
    processor = ONNXProcessor(
        persist_directory="./chroma_db_light",
        collection_name="pdf_documents_light",
    )

    print(f"\n{'=' * 60}")
    print(f"RAG SYSTEM WITH OPENAI")
    print(f"Database: {processor.count()} chunks indexed")
    print(f"{'=' * 60}\n")

    # Get question
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = input("Enter your question: ")

    print(f"\nQuestion: {question}\n")
    print("Retrieving relevant documents...")

    # Answer question
    result = answer_question_with_llm(question, processor, top_k=5)

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


if __name__ == "__main__":
    main()

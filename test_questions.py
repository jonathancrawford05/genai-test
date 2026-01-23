#!/usr/bin/env python3
"""
Test RAG system with questions from questions.csv
"""
import csv
import os
import time
from dotenv import load_dotenv
from openai import OpenAI

from src.onnx_processor import ONNXProcessor


def answer_question_with_llm(question: str, processor: ONNXProcessor, client: OpenAI, top_k: int = 5) -> dict:
    """Answer a question using RAG with OpenAI."""
    # Retrieve relevant chunks
    results = processor.query(question, top_k=top_k)

    if not results["documents"][0]:
        return {
            "answer": "I couldn't find any relevant information to answer this question.",
            "sources": [],
        }

    # Build context
    context_parts = []
    for i, doc in enumerate(results["documents"][0], 1):
        meta = results["metadatas"][0][i - 1]
        context_parts.append(
            f"[Source {i}] {meta['source_file']} (Page {meta['page_number']}):\n{doc}"
        )

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
- Do not make up information or use outside knowledge"""

    user_prompt = f"""Context from relevant documents:

{context}

Question: {question}

Answer:"""

    # Call OpenAI
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        max_tokens=1024,
    )

    return response.choices[0].message.content.strip()


def main():
    load_dotenv()

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not found in environment")
        sys.exit(1)

    # Initialize
    processor = ONNXProcessor(
        persist_directory="./chroma_db_light",
        collection_name="pdf_documents_light",
    )

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    print(f"\n{'=' * 60}")
    print(f"TESTING RAG SYSTEM WITH QUESTIONS.CSV")
    print(f"Database: {processor.count()} chunks indexed")
    print(f"{'=' * 60}\n")

    # Load questions
    questions_data = []
    with open("artifacts/questions.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            questions_data.append(row)

    print(f"Loaded {len(questions_data)} questions\n")

    # Answer each question
    results = []
    start_time = time.time()

    for i, row in enumerate(questions_data, 1):
        question_id = row["id"]
        question = row["question"]
        expected = row.get("expected_output", "")

        print(f"\n{'-' * 60}")
        print(f"[{i}/{len(questions_data)}] Question {question_id}")
        print(f"{'-' * 60}")
        print(f"Q: {question}\n")

        # Get answer
        answer = answer_question_with_llm(question, processor, client, top_k=5)

        print(f"A: {answer}\n")

        if expected:
            print(f"Expected: {expected[:200]}...")

        results.append({
            "id": question_id,
            "question": question,
            "answer": answer,
            "expected": expected,
        })

    elapsed = time.time() - start_time

    # Summary
    print(f"\n{'=' * 60}")
    print(f"COMPLETED")
    print(f"{'=' * 60}")
    print(f"Questions answered: {len(results)}")
    print(f"Time: {elapsed:.2f}s")
    print(f"Avg time per question: {elapsed / len(results):.2f}s")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()

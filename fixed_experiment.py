#!/usr/bin/env python3
"""
FIXED experimentation harness that actually uses different embeddings.

Properly supports:
- onnx: ChromaDB default ONNX embeddings
- sentence-transformers: PyTorch-based semantic embeddings
"""
import csv
import json
import time
import re
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import List, Dict
import ollama

from src.ultra_light_processor import UltraLightProcessor
from src.sentence_transformer_processor import SentenceTransformerProcessor


@dataclass
class DetailedQuestionResult:
    """Extended results with retrieval details."""
    question_id: str
    question: str
    expected: str
    answer: str
    exact_match: bool
    token_f1: float
    latency_sec: float
    config: str
    embedding_model: str
    llm_model: str
    llm_size_gb: float
    top_k: int
    retrieved_chunks: List[str] = field(default_factory=list)
    chunk_sources: List[str] = field(default_factory=list)
    chunk_scores: List[float] = field(default_factory=list)


class FixedRAGHarness:
    """Fixed harness that properly swaps embeddings."""

    MODEL_SIZES = {
        "phi3": 2.2,
        "llama3.2": 2.0,
        "llama3": 4.7,
        "llama3.1": 4.9,
        "qwen3:14b": 9.3,
        "gpt-oss:20b": 13.0,
    }

    def __init__(self, pdf_folder: str = "artifacts/1"):
        self.pdf_folder = pdf_folder
        self.processors = {}

    def get_or_create_processor(self, embedding_model: str, chunk_size: int, chunk_overlap: int):
        """Get or create processor - PROPERLY switching based on embedding type."""
        key = f"{embedding_model}_{chunk_size}_{chunk_overlap}"

        if key not in self.processors:
            print(f"\n{'=' * 60}")
            print(f"Setting up embedding: {embedding_model}")
            print(f"{'=' * 60}")

            # FIXED: Actually switch based on embedding type
            if embedding_model == "onnx":
                print("Using: ChromaDB ONNX embeddings (lightweight)")
                processor = UltraLightProcessor(
                    persist_directory=f"./chroma_db_onnx",
                    collection_name=f"pdf_docs_{key}",
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )

            elif embedding_model == "sentence-transformers":
                print("Using: Sentence-Transformers PyTorch embeddings (semantic)")
                processor = SentenceTransformerProcessor(
                    persist_directory=f"./chroma_db_sentence_transformers",
                    collection_name=f"pdf_docs_{key}",
                    model_name="all-MiniLM-L6-v2",
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )

            else:
                # For nomic-embed-text, we'd need Ollama integration
                # For now, fall back to ONNX with a warning
                print(f"⚠️  {embedding_model} not fully implemented, using ONNX")
                processor = UltraLightProcessor(
                    persist_directory=f"./chroma_db_{embedding_model.replace(':', '_')}",
                    collection_name=f"pdf_docs_{key}",
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )

            # Index if needed
            if processor.count() == 0:
                print(f"Indexing PDFs...")
                start = time.time()
                total = processor.process_folder(self.pdf_folder)
                elapsed = time.time() - start
                print(f"✓ Indexed {total} chunks in {elapsed:.1f}s")
                print(f"  (Indexing time is a good indicator - sentence-transformers should be 2-3x slower)")
            else:
                print(f"✓ Using existing index: {processor.count()} chunks")

            self.processors[key] = processor

        return self.processors[key]

    def query_with_ollama_detailed(self, question: str, processor, llm_model: str, top_k: int):
        """Query with detailed tracking."""
        start_time = time.time()

        # Retrieve
        results = processor.query(question, top_k=top_k)

        if not results["documents"][0]:
            return {
                "answer": "No relevant information found.",
                "latency": time.time() - start_time,
                "chunks": [],
                "sources": [],
                "scores": []
            }

        # Track retrieved chunks
        retrieved_chunks = results["documents"][0]
        chunk_sources = [
            f"{meta['source_file']} (p{meta['page_number']})"
            for meta in results["metadatas"][0]
        ]
        chunk_scores = [1 - d for d in results["distances"][0]]

        # Build context
        context_parts = []
        for i, doc in enumerate(retrieved_chunks, 1):
            meta = results["metadatas"][0][i - 1]
            context_parts.append(
                f"[Source {i}] {meta['source_file']} (Page {meta['page_number']}):\n{doc}"
            )
        context = "Context:\n\n" + "\n\n".join(context_parts)

        # Prompt
        system_prompt = """Answer questions using ONLY the provided context.
- Be precise and accurate
- For lists, use * bullets
- For calculations, provide just the number with $ if currency
- If insufficient info, say so"""

        user_prompt = f"{context}\n\nQuestion: {question}\n\nAnswer:"

        # Call Ollama
        response = ollama.chat(
            model=llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            options={"temperature": 0, "num_predict": 512}
        )

        latency = time.time() - start_time

        return {
            "answer": response["message"]["content"].strip(),
            "latency": latency,
            "chunks": [c[:200] + "..." if len(c) > 200 else c for c in retrieved_chunks],
            "sources": chunk_sources,
            "scores": chunk_scores
        }

    def evaluate_answer(self, answer: str, expected: str) -> tuple[bool, float]:
        """Evaluate answer quality."""
        def normalize(text: str) -> str:
            text = re.sub(r'\s+', ' ', text.lower()).strip()
            return re.sub(r'[^a-z0-9$ ]', '', text)

        norm_answer = normalize(answer)
        norm_expected = normalize(expected)

        exact = norm_answer == norm_expected

        # Token F1
        answer_tokens = set(norm_answer.split())
        expected_tokens = set(norm_expected.split())

        if not answer_tokens or not expected_tokens:
            f1 = 0.0
        else:
            common = answer_tokens & expected_tokens
            precision = len(common) / len(answer_tokens) if answer_tokens else 0
            recall = len(common) / len(expected_tokens) if expected_tokens else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return exact, f1

    def run_experiment_detailed(self, config: dict, questions_csv: str) -> List[DetailedQuestionResult]:
        """Run experiment with detailed tracking."""
        embedding = config["embedding"]
        llm = config["llm"]
        top_k = config["top_k"]

        print(f"\n{'#' * 60}")
        print(f"EXPERIMENT: {embedding}_{llm}_k{top_k}")
        print(f"{'#' * 60}\n")

        processor = self.get_or_create_processor(
            embedding,
            config.get("chunk_size", 2000),
            config.get("chunk_overlap", 200)
        )

        # Load questions
        questions = []
        with open(questions_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                questions.append(row)

        print(f"\nTesting {len(questions)} questions...\n")

        results = []
        for i, q in enumerate(questions, 1):
            print(f"[{i}/{len(questions)}] {q['id']}: ", end="", flush=True)

            response = self.query_with_ollama_detailed(q['question'], processor, llm, top_k)
            exact, f1 = self.evaluate_answer(response["answer"], q.get('expected_output', ''))

            result = DetailedQuestionResult(
                question_id=q['id'],
                question=q['question'],
                expected=q.get('expected_output', ''),
                answer=response["answer"],
                exact_match=exact,
                token_f1=f1,
                latency_sec=response["latency"],
                config=f"{embedding}_{llm}_k{top_k}",
                embedding_model=embedding,
                llm_model=llm,
                llm_size_gb=self.MODEL_SIZES.get(llm, 0),
                top_k=top_k,
                retrieved_chunks=response["chunks"],
                chunk_sources=response["sources"],
                chunk_scores=response["scores"]
            )

            results.append(result)

            status = "✓" if exact else f"F1={f1:.2f}"
            print(f"{status} ({response['latency']:.2f}s)")

        return results

    def export_detailed_csv(self, results: List[DetailedQuestionResult], output_file: str):
        """Export results to detailed CSV."""
        rows = []
        for r in results:
            row = {
                "question_id": r.question_id,
                "config": r.config,
                "embedding_model": r.embedding_model,
                "llm_model": r.llm_model,
                "top_k": r.top_k,
                "exact_match": r.exact_match,
                "token_f1": r.token_f1,
                "latency_sec": r.latency_sec,
                "question": r.question,
                "expected": r.expected,
                "answer": r.answer,
            }

            # Add chunk details
            for i in range(r.top_k):
                if i < len(r.retrieved_chunks):
                    row[f"chunk_{i+1}_source"] = r.chunk_sources[i]
                    row[f"chunk_{i+1}_score"] = f"{r.chunk_scores[i]:.3f}"
                    row[f"chunk_{i+1}_text"] = r.retrieved_chunks[i]
                else:
                    row[f"chunk_{i+1}_source"] = ""
                    row[f"chunk_{i+1}_score"] = ""
                    row[f"chunk_{i+1}_text"] = ""

            rows.append(row)

        if rows:
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

        print(f"\n✓ Detailed CSV exported to: {output_file}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="FIXED RAG Experimentation Harness")
    parser.add_argument('--embeddings', nargs='+', default=['onnx', 'sentence-transformers'],
                       help='Embedding models: onnx, sentence-transformers')
    parser.add_argument('--llms', nargs='+', default=['phi3', 'llama3.2'],
                       help='LLM models')
    parser.add_argument('--top-k', nargs='+', type=int, default=[5],
                       help='top_k values')
    parser.add_argument('--questions', default='artifacts/questions.csv')
    parser.add_argument('--output-csv', default='experiment_fixed_detailed.csv')

    args = parser.parse_args()

    harness = FixedRAGHarness()
    all_results = []

    # Run all combinations
    for embedding in args.embeddings:
        for llm in args.llms:
            for top_k in args.top_k:
                config = {"embedding": embedding, "llm": llm, "top_k": top_k}
                results = harness.run_experiment_detailed(config, args.questions)
                all_results.extend(results)

    # Export CSV
    harness.export_detailed_csv(all_results, args.output_csv)

    print(f"\n{'=' * 60}")
    print("✓ Fixed experiments complete!")
    print(f"  Results: {args.output_csv}")
    print(f"{'=' * 60}")
    print("\nNow check:")
    print("1. Indexing times - sentence-transformers should be 2-3x slower")
    print("2. F1 scores - should differ if embeddings retrieve different chunks")
    print("3. chunk_X_source columns - are they different across embeddings?")


if __name__ == "__main__":
    main()

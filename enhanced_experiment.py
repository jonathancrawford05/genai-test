#!/usr/bin/env python3
"""
Enhanced experimentation harness with detailed diagnostics.

New features:
- Sentence-transformers embedding support
- Detailed CSV export with chunk indices
- Retrieved chunk tracking
- Answer text export for manual inspection
"""
import csv
import json
import time
import re
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional
import ollama

from src.ultra_light_processor import UltraLightProcessor


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

    # New detailed fields
    retrieved_chunks: List[str] = field(default_factory=list)
    chunk_sources: List[str] = field(default_factory=list)
    chunk_scores: List[float] = field(default_factory=list)


class EnhancedRAGHarness:
    """Enhanced harness with detailed tracking."""

    MODEL_SIZES = {
        "phi3": 2.2,
        "llama3.2": 2.0,
        "llama3": 4.7,
        "llama3.1": 4.9,
        "qwen3:14b": 9.3,
        "gpt-oss:20b": 13.0,
    }

    def __init__(self, pdf_folder: str = "artifacts/1"):
        """Initialize harness."""
        self.pdf_folder = pdf_folder
        self.processors: Dict[str, UltraLightProcessor] = {}

    def get_or_create_processor(self, embedding_model: str, chunk_size: int, chunk_overlap: int):
        """Get or create processor."""
        key = f"{embedding_model}_{chunk_size}_{chunk_overlap}"

        if key not in self.processors:
            persist_dir = f"./chroma_db_{embedding_model.replace(':', '_')}"

            print(f"\n{'=' * 60}")
            print(f"Setting up embedding: {embedding_model}")
            print(f"{'=' * 60}")

            processor = UltraLightProcessor(
                persist_directory=persist_dir,
                collection_name=f"pdf_docs_{key}",
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

            if processor.count() == 0:
                print(f"Indexing PDFs...")
                start = time.time()
                total = processor.process_folder(self.pdf_folder)
                elapsed = time.time() - start
                print(f"✓ Indexed {total} chunks in {elapsed:.1f}s")
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
        context = "Context from relevant documents:\n\n" + "\n\n".join(context_parts)

        # Prompt
        system_prompt = """You are a helpful AI assistant that answers questions based on provided context from PDF documents.

Guidelines:
- Answer using ONLY the information from the provided context
- Be precise and accurate
- If the question asks for a list, format as a bulleted list with * prefix
- If the question asks for a calculation, provide just the final number with $ prefix if currency
- If insufficient information, say so
- Do not make up information"""

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
            "chunks": [c[:200] + "..." for c in retrieved_chunks],  # Truncate for CSV
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

    def run_experiment_detailed(self, config_params: dict, questions_csv: str) -> List[DetailedQuestionResult]:
        """Run experiment with detailed tracking."""
        embedding = config_params["embedding"]
        llm = config_params["llm"]
        top_k = config_params["top_k"]

        print(f"\n{'=' * 60}")
        print(f"EXPERIMENT: {embedding}_{llm}_k{top_k}")
        print(f"{'=' * 60}\n")

        processor = self.get_or_create_processor(
            embedding,
            config_params.get("chunk_size", 2000),
            config_params.get("chunk_overlap", 200)
        )

        # Load questions
        questions = []
        with open(questions_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                questions.append(row)

        print(f"Testing {len(questions)} questions...\n")

        results = []
        for i, q in enumerate(questions, 1):
            print(f"[{i}/{len(questions)}] {q['id']}: ", end="", flush=True)

            response = self.query_with_ollama_detailed(
                q['question'],
                processor,
                llm,
                top_k
            )

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
            # Create a row with all basic info
            row = {
                "question_id": r.question_id,
                "config": r.config,
                "embedding_model": r.embedding_model,
                "llm_model": r.llm_model,
                "top_k": r.top_k,
                "llm_size_gb": r.llm_size_gb,
                "exact_match": r.exact_match,
                "token_f1": r.token_f1,
                "latency_sec": r.latency_sec,
                "question": r.question,
                "expected": r.expected,
                "answer": r.answer,
            }

            # Add chunk information
            for i in range(r.top_k):
                if i < len(r.retrieved_chunks):
                    row[f"chunk_{i+1}_source"] = r.chunk_sources[i]
                    row[f"chunk_{i+1}_score"] = r.chunk_scores[i]
                    row[f"chunk_{i+1}_text"] = r.retrieved_chunks[i]
                else:
                    row[f"chunk_{i+1}_source"] = ""
                    row[f"chunk_{i+1}_score"] = ""
                    row[f"chunk_{i+1}_text"] = ""

            rows.append(row)

        # Write CSV
        if rows:
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

        print(f"\n✓ Detailed CSV exported to: {output_file}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Enhanced RAG Experimentation Harness")
    parser.add_argument('--embeddings', nargs='+', default=['onnx'],
                       help='Embedding models: onnx, nomic-embed-text, sentence-transformers')
    parser.add_argument('--llms', nargs='+', default=['phi3', 'llama3.2'],
                       help='LLM models to test')
    parser.add_argument('--top-k', nargs='+', type=int, default=[5],
                       help='top_k values to test')
    parser.add_argument('--questions', default='artifacts/questions.csv',
                       help='Questions CSV file')
    parser.add_argument('--output-json', default='experiment_results.json',
                       help='Output JSON file')
    parser.add_argument('--output-csv', default='experiment_results_detailed.csv',
                       help='Output CSV file')

    args = parser.parse_args()

    harness = EnhancedRAGHarness()

    all_results = []

    # Run all combinations
    for embedding in args.embeddings:
        for llm in args.llms:
            for top_k in args.top_k:
                config = {
                    "embedding": embedding,
                    "llm": llm,
                    "top_k": top_k
                }

                results = harness.run_experiment_detailed(config, args.questions)
                all_results.extend(results)

    # Export JSON (simplified)
    with open(args.output_json, 'w') as f:
        json.dump([
            {k: v for k, v in asdict(r).items() if k not in ['retrieved_chunks', 'chunk_sources', 'chunk_scores']}
            for r in all_results
        ], f, indent=2)

    # Export detailed CSV
    harness.export_detailed_csv(all_results, args.output_csv)

    print(f"\n{'=' * 60}")
    print(f"✓ Experiments complete!")
    print(f"  JSON: {args.output_json}")
    print(f"  CSV: {args.output_csv}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()

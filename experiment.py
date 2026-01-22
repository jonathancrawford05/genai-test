#!/usr/bin/env python3
"""
Experimentation harness for RAG PDF Q&A system.

Tests multiple combinations of:
- Embedding models (ultra-light ONNX, nomic-embed-text)
- LLM models (phi3, llama3.2, llama3.1, qwen3:14b, etc.)
- Retrieval parameters (top_k, chunk_size)

Tracks metrics:
- Accuracy (exact match, token F1)
- Latency (response time)
- Compute (model size as proxy)

Identifies optimal model for each question type.
"""
import csv
import json
import time
import re
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
import ollama

from src.ultra_light_processor import UltraLightProcessor


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    embedding_model: str  # "onnx" or "nomic-embed-text"
    llm_model: str  # e.g., "phi3", "llama3.2", "llama3.1", "qwen3:14b"
    top_k: int = 5
    chunk_size: int = 2000
    chunk_overlap: int = 200

    def __str__(self):
        return f"{self.embedding_model}_{self.llm_model}_k{self.top_k}"


@dataclass
class QuestionResult:
    """Results for a single question."""
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


class RAGExperimentHarness:
    """Harness for running RAG experiments."""

    # Model sizes (from your ollama list)
    MODEL_SIZES = {
        "phi3": 2.2,
        "llama3.2": 2.0,
        "llama3": 4.7,
        "llama3.1": 4.9,
        "qwen3:14b": 9.3,
        "gpt-oss:20b": 13.0,
        "nomic-embed-text": 0.274,
    }

    def __init__(self, pdf_folder: str = "artifacts/1"):
        """Initialize harness."""
        self.pdf_folder = pdf_folder
        self.processors: Dict[str, UltraLightProcessor] = {}

    def get_or_create_processor(self, embedding_model: str, chunk_size: int, chunk_overlap: int) -> UltraLightProcessor:
        """Get or create a processor for given embedding config."""
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

            # Check if needs indexing
            if processor.count() == 0:
                print(f"Indexing PDFs with {embedding_model}...")
                start = time.time()
                total = processor.process_folder(self.pdf_folder)
                elapsed = time.time() - start
                print(f"✓ Indexed {total} chunks in {elapsed:.1f}s")
            else:
                print(f"✓ Using existing index: {processor.count()} chunks")

            self.processors[key] = processor

        return self.processors[key]

    def query_with_ollama(self, question: str, processor: UltraLightProcessor,
                         llm_model: str, top_k: int) -> tuple[str, float]:
        """Query using Ollama LLM."""
        start_time = time.time()

        # Retrieve
        results = processor.query(question, top_k=top_k)

        if not results["documents"][0]:
            return "No relevant information found.", time.time() - start_time

        # Build context
        context_parts = []
        for i, doc in enumerate(results["documents"][0], 1):
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
        return response["message"]["content"].strip(), latency

    def evaluate_answer(self, answer: str, expected: str) -> tuple[bool, float]:
        """Evaluate answer quality."""
        # Normalize
        def normalize(text: str) -> str:
            text = re.sub(r'\s+', ' ', text.lower()).strip()
            return re.sub(r'[^a-z0-9$ ]', '', text)

        norm_answer = normalize(answer)
        norm_expected = normalize(expected)

        # Exact match
        exact = norm_answer == norm_expected

        # Token F1
        answer_tokens = set(norm_answer.split())
        expected_tokens = set(norm_expected.split())

        if not answer_tokens or not expected_tokens:
            f1 = 0.0
        else:
            common = answer_tokens & expected_tokens
            precision = len(common) / len(answer_tokens)
            recall = len(common) / len(expected_tokens)
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return exact, f1

    def run_experiment(self, config: ExperimentConfig, questions_csv: str) -> List[QuestionResult]:
        """Run experiment with given config."""
        print(f"\n{'=' * 60}")
        print(f"EXPERIMENT: {config}")
        print(f"{'=' * 60}\n")

        # Get processor
        processor = self.get_or_create_processor(
            config.embedding_model,
            config.chunk_size,
            config.chunk_overlap
        )

        # Load questions
        questions = []
        with open(questions_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                questions.append(row)

        print(f"Testing {len(questions)} questions with {config.llm_model}...\n")

        # Answer each question
        results = []
        for i, q in enumerate(questions, 1):
            print(f"[{i}/{len(questions)}] {q['id']}: ", end="", flush=True)

            answer, latency = self.query_with_ollama(
                q['question'],
                processor,
                config.llm_model,
                config.top_k
            )

            exact, f1 = self.evaluate_answer(answer, q.get('expected_output', ''))

            result = QuestionResult(
                question_id=q['id'],
                question=q['question'],
                expected=q.get('expected_output', ''),
                answer=answer,
                exact_match=exact,
                token_f1=f1,
                latency_sec=latency,
                config=str(config),
                embedding_model=config.embedding_model,
                llm_model=config.llm_model,
                llm_size_gb=self.MODEL_SIZES.get(config.llm_model, 0),
                top_k=config.top_k
            )

            results.append(result)

            status = "✓" if exact else f"F1={f1:.2f}"
            print(f"{status} ({latency:.2f}s)")

        return results

    def run_grid_search(self,
                       embedding_models: List[str],
                       llm_models: List[str],
                       top_k_values: List[int],
                       questions_csv: str,
                       output_file: str = "experiment_results.json") -> List[QuestionResult]:
        """Run experiments across all combinations."""
        all_results = []

        total_experiments = len(embedding_models) * len(llm_models) * len(top_k_values)
        exp_num = 0

        for embedding in embedding_models:
            for llm in llm_models:
                for top_k in top_k_values:
                    exp_num += 1
                    print(f"\n{'#' * 60}")
                    print(f"EXPERIMENT {exp_num}/{total_experiments}")
                    print(f"{'#' * 60}")

                    config = ExperimentConfig(
                        embedding_model=embedding,
                        llm_model=llm,
                        top_k=top_k
                    )

                    results = self.run_experiment(config, questions_csv)
                    all_results.extend(results)

        # Save results
        with open(output_file, 'w') as f:
            json.dump([asdict(r) for r in all_results], f, indent=2)

        print(f"\n{'=' * 60}")
        print(f"✓ All experiments complete!")
        print(f"  Results saved to: {output_file}")
        print(f"{'=' * 60}\n")

        return all_results

    def analyze_results(self, results: List[QuestionResult]):
        """Analyze and print results summary."""
        print("\n" + "=" * 80)
        print("RESULTS SUMMARY")
        print("=" * 80)

        # Group by config
        by_config = {}
        for r in results:
            key = f"{r.embedding_model}_{r.llm_model}_k{r.top_k}"
            if key not in by_config:
                by_config[key] = []
            by_config[key].append(r)

        # Compute aggregates
        print(f"\n{'Config':<40} {'Exact':<8} {'Avg F1':<8} {'Avg Lat':<10} {'Size(GB)':<10}")
        print("-" * 80)

        for config_name, config_results in sorted(by_config.items()):
            exact_count = sum(1 for r in config_results if r.exact_match)
            avg_f1 = sum(r.token_f1 for r in config_results) / len(config_results)
            avg_lat = sum(r.latency_sec for r in config_results) / len(config_results)
            size = config_results[0].llm_size_gb

            print(f"{config_name:<40} {exact_count}/{len(config_results):<6} "
                  f"{avg_f1:.3f}    {avg_lat:.2f}s      {size:.1f}")

        # Best performers
        print("\n" + "=" * 80)
        print("BEST PERFORMERS")
        print("=" * 80)

        # Best accuracy
        best_f1 = max(by_config.items(),
                     key=lambda x: sum(r.token_f1 for r in x[1]) / len(x[1]))
        print(f"\nBest Accuracy: {best_f1[0]}")
        print(f"  Avg F1: {sum(r.token_f1 for r in best_f1[1]) / len(best_f1[1]):.3f}")

        # Fastest
        fastest = min(by_config.items(),
                     key=lambda x: sum(r.latency_sec for r in x[1]) / len(x[1]))
        print(f"\nFastest: {fastest[0]}")
        print(f"  Avg latency: {sum(r.latency_sec for r in fastest[1]) / len(fastest[1]):.2f}s")

        # Best efficiency (F1 / latency)
        best_eff = max(by_config.items(),
                      key=lambda x: (sum(r.token_f1 for r in x[1]) / len(x[1])) /
                                   (sum(r.latency_sec for r in x[1]) / len(x[1])))
        print(f"\nBest Efficiency (F1/latency): {best_eff[0]}")

        # Per-question analysis
        print("\n" + "=" * 80)
        print("PER-QUESTION OPTIMAL MODEL")
        print("=" * 80)

        questions = list(set(r.question_id for r in results))
        for qid in sorted(questions):
            q_results = [r for r in results if r.question_id == qid]

            # Best for this question
            best = max(q_results, key=lambda r: r.token_f1)

            # Smallest model with same performance
            same_perf = [r for r in q_results if r.token_f1 >= best.token_f1 * 0.95]
            smallest = min(same_perf, key=lambda r: r.llm_size_gb)

            print(f"\n{qid}:")
            print(f"  Best: {best.llm_model} (F1={best.token_f1:.3f}, {best.latency_sec:.2f}s, {best.llm_size_gb}GB)")
            if smallest.llm_model != best.llm_model:
                print(f"  Optimal: {smallest.llm_model} (F1={smallest.token_f1:.3f}, "
                      f"{smallest.latency_sec:.2f}s, {smallest.llm_size_gb}GB) ← Use this!")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="RAG Experimentation Harness")
    parser.add_argument('--embeddings', nargs='+', default=['onnx', 'nomic-embed-text'],
                       help='Embedding models to test')
    parser.add_argument('--llms', nargs='+', default=['phi3', 'llama3.2', 'llama3.1'],
                       help='LLM models to test')
    parser.add_argument('--top-k', nargs='+', type=int, default=[5],
                       help='top_k values to test')
    parser.add_argument('--questions', default='artifacts/questions.csv',
                       help='Questions CSV file')
    parser.add_argument('--output', default='experiment_results.json',
                       help='Output file for results')
    parser.add_argument('--quick', action='store_true',
                       help='Quick test (phi3 + llama3.2 only)')

    args = parser.parse_args()

    # Quick mode
    if args.quick:
        args.llms = ['phi3', 'llama3.2']
        args.embeddings = ['onnx']
        args.top_k = [5]

    # Run experiments
    harness = RAGExperimentHarness()

    results = harness.run_grid_search(
        embedding_models=args.embeddings,
        llm_models=args.llms,
        top_k_values=args.top_k,
        questions_csv=args.questions,
        output_file=args.output
    )

    # Analyze
    harness.analyze_results(results)


if __name__ == "__main__":
    main()

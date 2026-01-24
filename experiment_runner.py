"""
Experimentation harness for evaluating PDF QA system variations.

Part 2 of GenAI Test: Systematic evaluation of different configurations.
"""
import pandas as pd
import time
import json
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

from answer_pdf_question import answer_pdf_question
from src.agents import OrchestratorConfig, RouterConfig, PlannerConfig, RetrieverConfig


@dataclass
class VariationConfig:
    """Configuration for a single experiment variation."""
    name: str
    description: str
    orchestrator_config: OrchestratorConfig


@dataclass
class QuestionResult:
    """Result from answering a single question."""
    question_id: str
    question: str
    expected_output: str
    actual_output: str
    variation_name: str

    # Metrics
    exact_match: bool
    partial_match_score: float  # 0.0 - 1.0
    answer_length: int
    execution_time_seconds: float

    # Retrieval stats (if available)
    documents_selected: Optional[int] = None
    chunks_retrieved: Optional[int] = None

    # Failure analysis
    error_occurred: bool = False
    error_message: Optional[str] = None


class ExperimentRunner:
    """
    Runs experiments across multiple variations and test questions.

    Implements Part 2 of GenAI test: systematic evaluation framework.
    """

    def __init__(self, questions_csv: str = "artifacts/questions.csv"):
        """
        Initialize experiment runner.

        Args:
            questions_csv: Path to questions CSV file
        """
        self.questions_df = pd.read_csv(questions_csv)
        self.results: List[QuestionResult] = []

    def clean_chroma_collections(self):
        """
        Clean ChromaDB collections to avoid dimension mismatches.

        This ensures each variation starts with fresh, correctly-dimensioned collections.
        """
        import shutil

        dirs_to_clean = ["./chroma_db_onnx", "./chroma_db_ollama"]

        for dir_path in dirs_to_clean:
            if Path(dir_path).exists():
                print(f"  Cleaning {dir_path}...")
                shutil.rmtree(dir_path)

        print("✓ ChromaDB collections cleaned\n")

    def create_variations(self) -> List[VariationConfig]:
        """
        Create experiment variations to test.

        Returns:
            List of variation configurations
        """
        variations = [
            # Variation 1: Baseline (current system)
            VariationConfig(
                name="baseline",
                description="ONNX embeddings, pre-filtered search, top_k_docs=3, top_k_per_step=5",
                orchestrator_config=OrchestratorConfig(
                    model="llama3.2",
                    temperature=0.0,
                    max_answer_tokens=2048,
                    router_config=RouterConfig(
                        model="llama3.2",
                        top_k_docs=3,
                        temperature=0.0
                    ),
                    planner_config=PlannerConfig(
                        model="llama3.2",
                        temperature=0.0
                    ),
                    retriever_config=RetrieverConfig(
                        embedding_type="onnx",
                        top_k_per_step=5
                    )
                )
            ),

            # Variation 2: Ollama embeddings (higher quality)
            VariationConfig(
                name="ollama_embeddings",
                description="Ollama embeddings (nomic-embed-text), pre-filtered search, same parameters",
                orchestrator_config=OrchestratorConfig(
                    model="llama3.2",
                    temperature=0.0,
                    max_answer_tokens=2048,
                    router_config=RouterConfig(
                        model="llama3.2",
                        top_k_docs=3,
                        temperature=0.0
                    ),
                    planner_config=PlannerConfig(
                        model="llama3.2",
                        temperature=0.0
                    ),
                    retriever_config=RetrieverConfig(
                        embedding_type="ollama",
                        top_k_per_step=5
                    )
                )
            ),

            # Variation 3: Higher retrieval depth
            VariationConfig(
                name="high_depth",
                description="ONNX embeddings, top_k_docs=5, top_k_per_step=10 (more context)",
                orchestrator_config=OrchestratorConfig(
                    model="llama3.2",
                    temperature=0.0,
                    max_answer_tokens=2048,
                    router_config=RouterConfig(
                        model="llama3.2",
                        top_k_docs=5,  # More documents
                        temperature=0.0
                    ),
                    planner_config=PlannerConfig(
                        model="llama3.2",
                        temperature=0.0
                    ),
                    retriever_config=RetrieverConfig(
                        embedding_type="onnx",
                        top_k_per_step=10  # More chunks per step
                    )
                )
            ),

            # Variation 4: Conservative retrieval
            VariationConfig(
                name="conservative",
                description="ONNX embeddings, top_k_docs=2, top_k_per_step=3 (less context, faster)",
                orchestrator_config=OrchestratorConfig(
                    model="llama3.2",
                    temperature=0.0,
                    max_answer_tokens=2048,
                    router_config=RouterConfig(
                        model="llama3.2",
                        top_k_docs=2,  # Fewer documents
                        temperature=0.0
                    ),
                    planner_config=PlannerConfig(
                        model="llama3.2",
                        temperature=0.0
                    ),
                    retriever_config=RetrieverConfig(
                        embedding_type="onnx",
                        top_k_per_step=3  # Fewer chunks per step
                    )
                )
            ),
        ]

        return variations

    def run_single_question(
        self,
        question_id: str,
        question: str,
        expected_output: str,
        pdf_folder: str,
        variation: VariationConfig,
        verbose: bool = False
    ) -> QuestionResult:
        """
        Run a single question with a specific variation.

        Args:
            question_id: Question identifier
            question: Question text
            expected_output: Expected answer
            pdf_folder: Path to PDF folder
            variation: Variation configuration
            verbose: Print progress

        Returns:
            QuestionResult with metrics
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f"Question: {question_id}")
            print(f"Variation: {variation.name}")
            print(f"{'='*70}")

        start_time = time.time()
        error_occurred = False
        error_message = None
        actual_output = ""

        try:
            # Run the answer function with this variation's config
            # Note: We'll need to modify answer_pdf_question to accept config
            # For now, we'll use a wrapper approach
            from src.agents import OrchestratorAgent

            orchestrator = OrchestratorAgent(
                summaries_path="artifacts/document_summaries.json",
                pdf_folder=f"artifacts/{pdf_folder}",
                config=variation.orchestrator_config
            )

            result = orchestrator.answer_question(question, verbose=verbose)
            actual_output = result.answer

        except Exception as e:
            error_occurred = True
            error_message = str(e)
            actual_output = f"ERROR: {error_message}"
            if verbose:
                print(f"\n❌ Error: {error_message}")

        execution_time = time.time() - start_time

        # Calculate metrics
        exact_match = self._check_exact_match(actual_output, expected_output)
        partial_match = self._calculate_partial_match(actual_output, expected_output)

        result = QuestionResult(
            question_id=question_id,
            question=question,
            expected_output=expected_output,
            actual_output=actual_output,
            variation_name=variation.name,
            exact_match=exact_match,
            partial_match_score=partial_match,
            answer_length=len(actual_output),
            execution_time_seconds=round(execution_time, 2),
            error_occurred=error_occurred,
            error_message=error_message
        )

        if verbose:
            print(f"\n{'─'*70}")
            print(f"Results:")
            print(f"  Exact match: {exact_match}")
            print(f"  Partial match: {partial_match:.2%}")
            print(f"  Execution time: {execution_time:.2f}s")
            print(f"{'─'*70}")

        return result

    def run_all_experiments(self, verbose: bool = True, clean_collections: bool = True) -> pd.DataFrame:
        """
        Run all variations on all questions.

        Args:
            verbose: Print progress
            clean_collections: Clean ChromaDB collections before starting (recommended)

        Returns:
            DataFrame with all results
        """
        variations = self.create_variations()

        print(f"\n{'='*70}")
        print(f"EXPERIMENT RUNNER - Part 2")
        print(f"{'='*70}")
        print(f"Questions: {len(self.questions_df)}")
        print(f"Variations: {len(variations)}")
        print(f"Total runs: {len(self.questions_df) * len(variations)}")
        print(f"{'='*70}\n")

        # Clean collections to avoid dimension mismatches
        if clean_collections:
            print("Cleaning ChromaDB collections to avoid dimension mismatches...")
            self.clean_chroma_collections()

        for idx, row in self.questions_df.iterrows():
            question_id = row['id']
            question = row['question']
            expected_output = row['expected_output']
            pdf_folder = row['PDF Folder']

            for variation in variations:
                result = self.run_single_question(
                    question_id=question_id,
                    question=question,
                    expected_output=expected_output,
                    pdf_folder=pdf_folder,
                    variation=variation,
                    verbose=verbose
                )

                self.results.append(result)

        # Convert to DataFrame
        results_df = pd.DataFrame([asdict(r) for r in self.results])

        return results_df

    def _check_exact_match(self, actual: str, expected: str) -> bool:
        """
        Check if actual matches expected exactly (case-insensitive, whitespace normalized).

        Args:
            actual: Actual answer
            expected: Expected answer

        Returns:
            True if exact match
        """
        actual_norm = " ".join(actual.lower().split())
        expected_norm = " ".join(expected.lower().split())

        return actual_norm == expected_norm

    def _calculate_partial_match(self, actual: str, expected: str) -> float:
        """
        Calculate partial match score based on keyword overlap.

        For EF_1 (list questions): Count how many expected items appear in actual
        For EF_2 (numeric): Check if expected number appears in actual

        Args:
            actual: Actual answer
            expected: Expected answer

        Returns:
            Score between 0.0 and 1.0
        """
        # Normalize
        actual_lower = actual.lower()
        expected_lower = expected.lower()

        # Check if it's a list (contains bullet points or multiple lines)
        if '*' in expected or '\n' in expected:
            # It's a list - count how many items are present
            expected_items = [
                line.strip().strip('*').strip()
                for line in expected.split('\n')
                if line.strip() and line.strip() != '*'
            ]

            matches = sum(
                1 for item in expected_items
                if item.lower() in actual_lower and len(item) > 3
            )

            total = len(expected_items)
            return matches / total if total > 0 else 0.0

        else:
            # It's likely a number or short answer
            # Check if expected substring appears in actual
            if expected_lower in actual_lower:
                return 1.0

            # Extract numbers and compare
            import re
            expected_numbers = set(re.findall(r'\d+', expected))
            actual_numbers = set(re.findall(r'\d+', actual))

            if expected_numbers and actual_numbers:
                overlap = len(expected_numbers & actual_numbers)
                return overlap / len(expected_numbers)

            return 0.0

    def generate_report(self, results_df: pd.DataFrame, output_dir: str = "results"):
        """
        Generate summary report and save results.

        Args:
            results_df: Results DataFrame
            output_dir: Directory to save results
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save detailed results
        csv_path = output_path / f"experiment_results_{timestamp}.csv"
        results_df.to_csv(csv_path, index=False)
        print(f"\n✓ Saved detailed results: {csv_path}")

        # Generate summary
        summary = self._generate_summary(results_df)

        # Save summary report
        report_path = output_path / f"summary_report_{timestamp}.md"
        with open(report_path, 'w') as f:
            f.write(summary)
        print(f"✓ Saved summary report: {report_path}")

        # Print summary to console
        print(f"\n{'='*70}")
        print("EXPERIMENT SUMMARY")
        print(f"{'='*70}\n")
        print(summary)

        return summary

    def _generate_summary(self, results_df: pd.DataFrame) -> str:
        """
        Generate markdown summary of results.

        Args:
            results_df: Results DataFrame

        Returns:
            Markdown summary string
        """
        summary_parts = []

        summary_parts.append("# Experiment Results Summary\n")
        summary_parts.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        summary_parts.append(f"Total experiments: {len(results_df)}\n")

        # Group by variation
        summary_parts.append("\n## Results by Variation\n")

        for variation_name in results_df['variation_name'].unique():
            var_df = results_df[results_df['variation_name'] == variation_name]

            summary_parts.append(f"\n### {variation_name}\n")

            # Metrics
            exact_matches = var_df['exact_match'].sum()
            avg_partial = var_df['partial_match_score'].mean()
            avg_time = var_df['execution_time_seconds'].mean()
            errors = var_df['error_occurred'].sum()

            summary_parts.append(f"- **Exact matches:** {exact_matches}/{len(var_df)}")
            summary_parts.append(f"- **Average partial match:** {avg_partial:.1%}")
            summary_parts.append(f"- **Average execution time:** {avg_time:.2f}s")
            summary_parts.append(f"- **Errors:** {errors}/{len(var_df)}")

            # Per-question breakdown
            summary_parts.append("\n**Per-question results:**\n")
            for _, row in var_df.iterrows():
                status = "✅" if row['exact_match'] else f"🔶 {row['partial_match_score']:.0%}"
                summary_parts.append(
                    f"- {row['question_id']}: {status} ({row['execution_time_seconds']:.1f}s)"
                )

            summary_parts.append("")

        # Comparison table
        summary_parts.append("\n## Variation Comparison\n")
        summary_parts.append("| Variation | Exact Match | Avg Partial | Avg Time (s) | Errors |")
        summary_parts.append("|-----------|-------------|-------------|--------------|--------|")

        for variation_name in results_df['variation_name'].unique():
            var_df = results_df[results_df['variation_name'] == variation_name]
            exact = f"{var_df['exact_match'].sum()}/{len(var_df)}"
            partial = f"{var_df['partial_match_score'].mean():.1%}"
            time_avg = f"{var_df['execution_time_seconds'].mean():.2f}"
            errors = var_df['error_occurred'].sum()

            summary_parts.append(f"| {variation_name} | {exact} | {partial} | {time_avg} | {errors} |")

        # Recommendations
        summary_parts.append("\n## Recommendations\n")

        best_accuracy_var = results_df.groupby('variation_name')['partial_match_score'].mean().idxmax()
        best_speed_var = results_df.groupby('variation_name')['execution_time_seconds'].mean().idxmin()

        summary_parts.append(f"- **Best accuracy:** {best_accuracy_var}")
        summary_parts.append(f"- **Fastest:** {best_speed_var}")

        return "\n".join(summary_parts)


# Main execution
if __name__ == "__main__":
    runner = ExperimentRunner()
    results_df = runner.run_all_experiments(verbose=True)
    runner.generate_report(results_df)

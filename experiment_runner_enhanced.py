"""
Enhanced experimentation harness with detailed JSON output for observability.

Captures full configuration, retrieved chunk details, and intermediate agent decisions
for deep analysis and reproducibility.
"""
import pandas as pd
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

from answer_pdf_question import answer_pdf_question
from src.agents import (
    OrchestratorConfig, RouterConfig, PlannerConfig, RetrieverConfig,
    OrchestratorAgent
)


@dataclass
class VariationConfig:
    """Configuration for a single experiment variation."""
    name: str
    description: str
    orchestrator_config: OrchestratorConfig


@dataclass
class DetailedQuestionResult:
    """Enhanced result with full context for analysis."""
    # Basic info
    question_id: str
    question: str
    expected_output: str
    actual_output: str
    variation_name: str

    # Configuration (for reproducibility)
    config: Dict[str, Any]

    # Router decisions
    router_selected_docs: List[str]

    # Planner decisions
    planner_strategy: str
    planner_num_steps: int
    planner_requires_combination: bool

    # Retriever details
    retrieval_steps: List[Dict[str, Any]]  # Step-by-step retrieval with chunk details
    total_chunks_retrieved: int

    # Metrics
    exact_match: bool
    partial_match_score: float
    answer_length: int
    execution_time_seconds: float

    # Failure analysis
    error_occurred: bool = False
    error_message: Optional[str] = None


class EnhancedExperimentRunner:
    """
    Runs experiments with detailed observability and JSON output.

    Captures:
    - Full configuration for reproducibility
    - Retrieved chunk indices, sources, pages, distances
    - Router/Planner decisions
    - Step-by-step retrieval details
    """

    def __init__(self, questions_csv: str = "artifacts/questions.csv"):
        """
        Initialize enhanced experiment runner.

        Args:
            questions_csv: Path to questions CSV file
        """
        self.questions_df = pd.read_csv(questions_csv)
        self.results: List[DetailedQuestionResult] = []

    def clean_chroma_collections(self):
        """Clean ChromaDB collections to ensure fresh indexes."""
        import shutil

        dir_path = "./chroma_db"

        if Path(dir_path).exists():
            print(f"  Cleaning {dir_path}...")
            shutil.rmtree(dir_path)

        print("✓ ChromaDB collections cleaned\n")

    def create_variations(self) -> List[VariationConfig]:
        """
        Create experiment variations to test.

        Override this method to define custom variations.

        Returns:
            List of variation configurations
        """
        # Default variations - can be overridden
        variations = [
            VariationConfig(
                name="baseline",
                description="Baseline configuration",
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
                        top_k_per_step=5,
                        chunk_size=1000,
                        chunk_overlap=200,
                        expand_context=0,
                        chunking_strategy="document"
                    )
                )
            ),
        ]

        return variations

    def run_single_question_detailed(
        self,
        question_id: str,
        question: str,
        expected_output: str,
        pdf_folder: str,
        variation: VariationConfig,
        verbose: bool = False
    ) -> DetailedQuestionResult:
        """
        Run a single question with detailed tracking.

        Args:
            question_id: Question identifier
            question: Question text
            expected_output: Expected answer
            pdf_folder: Path to PDF folder
            variation: Variation configuration
            verbose: Print progress

        Returns:
            DetailedQuestionResult with full context
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

        # Initialize tracking variables
        router_selected_docs = []
        planner_strategy = ""
        planner_num_steps = 0
        planner_requires_combination = False
        retrieval_steps = []
        total_chunks_retrieved = 0

        try:
            # Create orchestrator with this variation's config
            orchestrator = OrchestratorAgent(
                summaries_path="artifacts/document_summaries.json",
                pdf_folder=f"artifacts/{pdf_folder}",
                config=variation.orchestrator_config
            )

            # CAPTURE ROUTER DECISIONS
            selected_docs = orchestrator.router.select_documents(
                question=question,
                verbose=verbose
            )
            router_selected_docs = selected_docs

            # CAPTURE PLANNER DECISIONS
            plan = orchestrator.planner.create_plan(
                question=question,
                selected_documents=selected_docs,
                summaries=orchestrator.router.summaries,
                verbose=verbose
            )
            planner_strategy = plan.strategy
            planner_num_steps = len(plan.steps)
            planner_requires_combination = plan.requires_combination

            # CAPTURE RETRIEVER EXECUTION
            execution_result = orchestrator.retriever.execute_plan(
                plan=plan,
                verbose=verbose
            )

            # Extract detailed retrieval information
            for step_result in execution_result.step_results:
                step_info = {
                    "step_number": step_result.step_number,
                    "description": step_result.description,
                    "query": step_result.query,
                    "num_chunks": len(step_result.chunks),
                    "chunks": []
                }

                # Capture chunk details
                for chunk in step_result.chunks:
                    chunk_info = {
                        "chunk_index": chunk.get("chunk_index"),
                        "source_file": chunk.get("source_file"),
                        "page_number": chunk.get("page_number"),
                        "distance": chunk.get("distance"),
                        "text_preview": chunk.get("text", "")[:200] + "..." if len(chunk.get("text", "")) > 200 else chunk.get("text", ""),
                        "text_length": len(chunk.get("text", ""))
                    }
                    step_info["chunks"].append(chunk_info)

                retrieval_steps.append(step_info)

            total_chunks_retrieved = sum(len(r.chunks) for r in execution_result.step_results)

            # GENERATE ANSWER
            answer = orchestrator._generate_answer(
                question=question,
                execution_result=execution_result,
                verbose=verbose
            )
            actual_output = answer.answer

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

        # Extract full configuration
        config_dict = {
            "variation_name": variation.name,
            "description": variation.description,
            "model": variation.orchestrator_config.model,
            "temperature": variation.orchestrator_config.temperature,
            "router": {
                "top_k_docs": variation.orchestrator_config.router_config.top_k_docs,
            },
            "retriever": {
                "top_k_per_step": variation.orchestrator_config.retriever_config.top_k_per_step,
                "chunk_size": variation.orchestrator_config.retriever_config.chunk_size,
                "chunk_overlap": variation.orchestrator_config.retriever_config.chunk_overlap,
                "expand_context": variation.orchestrator_config.retriever_config.expand_context,
                "chunking_strategy": variation.orchestrator_config.retriever_config.chunking_strategy,
                "use_hybrid": variation.orchestrator_config.retriever_config.use_hybrid,
                "hybrid_alpha": variation.orchestrator_config.retriever_config.hybrid_alpha,
            }
        }

        result = DetailedQuestionResult(
            question_id=question_id,
            question=question,
            expected_output=expected_output,
            actual_output=actual_output,
            variation_name=variation.name,
            config=config_dict,
            router_selected_docs=router_selected_docs,
            planner_strategy=planner_strategy,
            planner_num_steps=planner_num_steps,
            planner_requires_combination=planner_requires_combination,
            retrieval_steps=retrieval_steps,
            total_chunks_retrieved=total_chunks_retrieved,
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
            print(f"  Router selected: {len(router_selected_docs)} docs")
            print(f"  Planner created: {planner_num_steps} step(s)")
            print(f"  Retrieved: {total_chunks_retrieved} chunks")
            print(f"  Exact match: {exact_match}")
            print(f"  Partial match: {partial_match:.2%}")
            print(f"  Execution time: {execution_time:.2f}s")
            print(f"{'─'*70}")

        return result

    def run_all_experiments(
        self,
        verbose: bool = True,
        clean_collections: bool = True,
        save_detailed_json: bool = True
    ) -> pd.DataFrame:
        """
        Run all variations on all questions with detailed tracking.

        Args:
            verbose: Print progress
            clean_collections: Clean ChromaDB collections before starting
            save_detailed_json: Save detailed JSON output

        Returns:
            DataFrame with all results
        """
        variations = self.create_variations()

        print(f"\n{'='*70}")
        print(f"ENHANCED EXPERIMENT RUNNER")
        print(f"{'='*70}")
        print(f"Questions: {len(self.questions_df)}")
        print(f"Variations: {len(variations)}")
        print(f"Total runs: {len(self.questions_df) * len(variations)}")
        print(f"{'='*70}\n")

        if clean_collections:
            print("Cleaning ChromaDB collections...")
            self.clean_chroma_collections()

        for idx, row in self.questions_df.iterrows():
            question_id = row['id']
            question = row['question']
            expected_output = row['expected_output']
            pdf_folder = row['PDF Folder']

            for variation in variations:
                result = self.run_single_question_detailed(
                    question_id=question_id,
                    question=question,
                    expected_output=expected_output,
                    pdf_folder=pdf_folder,
                    variation=variation,
                    verbose=verbose
                )

                self.results.append(result)

        # Convert to DataFrame (simplified version)
        results_dicts = []
        for r in self.results:
            result_dict = {
                "question_id": r.question_id,
                "question": r.question,
                "variation_name": r.variation_name,
                "exact_match": r.exact_match,
                "partial_match_score": r.partial_match_score,
                "execution_time_seconds": r.execution_time_seconds,
                "total_chunks_retrieved": r.total_chunks_retrieved,
                "router_selected_docs": len(r.router_selected_docs),
                "planner_num_steps": r.planner_num_steps,
                "error_occurred": r.error_occurred,
            }
            results_dicts.append(result_dict)

        results_df = pd.DataFrame(results_dicts)

        # Save detailed JSON
        if save_detailed_json:
            self.save_detailed_json()

        return results_df

    def save_detailed_json(self, output_dir: str = "results"):
        """
        Save detailed JSON with full context for reproducibility.

        Args:
            output_dir: Directory to save results
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = output_path / f"detailed_results_{timestamp}.json"

        # Convert results to JSON-serializable format
        detailed_results = {
            "experiment_metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_variations": len(set(r.variation_name for r in self.results)),
                "total_questions": len(set(r.question_id for r in self.results)),
                "total_runs": len(self.results)
            },
            "results": [asdict(r) for r in self.results]
        }

        with open(json_path, 'w') as f:
            json.dump(detailed_results, f, indent=2)

        print(f"\n✓ Saved detailed JSON: {json_path}")
        print(f"  Size: {json_path.stat().st_size / 1024:.1f} KB")
        print(f"  Contains: Full config, chunk indices, router/planner decisions\n")

    def _check_exact_match(self, actual: str, expected: str) -> bool:
        """Check if actual matches expected exactly."""
        actual_norm = " ".join(actual.lower().split())
        expected_norm = " ".join(expected.lower().split())
        return actual_norm == expected_norm

    def _calculate_partial_match(self, actual: str, expected: str) -> float:
        """Calculate partial match score based on keyword overlap."""
        actual_lower = actual.lower()
        expected_lower = expected.lower()

        if '*' in expected or '\n' in expected:
            # List question - count items
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
            # Short answer - check if expected appears in actual
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
        Generate summary report.

        Args:
            results_df: Results DataFrame
            output_dir: Directory to save results
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save CSV
        csv_path = output_path / f"experiment_results_{timestamp}.csv"
        results_df.to_csv(csv_path, index=False)
        print(f"✓ Saved results CSV: {csv_path}")

        # Generate and save summary
        from experiment_runner import ExperimentRunner
        runner = ExperimentRunner()
        summary = runner._generate_summary(results_df)

        report_path = output_path / f"summary_report_{timestamp}.md"
        with open(report_path, 'w') as f:
            f.write(summary)
        print(f"✓ Saved summary report: {report_path}")

        print(f"\n{'='*70}")
        print("EXPERIMENT SUMMARY")
        print(f"{'='*70}\n")
        print(summary)


# Main execution
if __name__ == "__main__":
    runner = EnhancedExperimentRunner()
    results_df = runner.run_all_experiments(verbose=True)
    runner.generate_report(results_df)

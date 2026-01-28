#!/usr/bin/env python3
"""
Component-based scoring for EF_2 question.

Evaluates answers based on:
1. Document Retrieval (20 points)
2. Deductible Identification (20 points)
3. Base Rate Identification (20 points)
4. Factor Identification (20 points)
5. Calculation (20 points)

Total: 100 points possible
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, asdict
import pandas as pd


@dataclass
class ComponentScore:
    """Score breakdown for each component."""
    document_retrieval: float = 0.0
    deductible_identification: float = 0.0
    base_rate_identification: float = 0.0
    factor_identification: float = 0.0
    calculation: float = 0.0

    @property
    def total(self) -> float:
        return (
            self.document_retrieval +
            self.deductible_identification +
            self.base_rate_identification +
            self.factor_identification +
            self.calculation
        )

    def to_dict(self) -> Dict[str, float]:
        return {
            'document_retrieval': self.document_retrieval,
            'deductible_identification': self.deductible_identification,
            'base_rate_identification': self.base_rate_identification,
            'factor_identification': self.factor_identification,
            'calculation': self.calculation,
            'total': self.total
        }


class EF2Scorer:
    """Scores EF_2 answers based on component criteria."""

    # Expected ground truth values
    REQUIRED_DOCS = {
        "rules_manual": "CT MAPS Homeowner Rules Manual",
        "rate_pages": "CT Homeowners MAPS Rate Pages"
    }

    DEDUCTIBLE_PERCENT = 2
    BASE_RATE = 293
    FACTOR = 2.061
    CORRECT_ANSWER = 604  # Or ~603.87

    # Keywords for detection
    DEDUCTIBLE_KEYWORDS = ["2%", "2 %", "2 percent", "two percent"]
    RULE_C7_KEYWORDS = ["rule c-7", "rule c7", "c-7", "2,500 feet", "2500 feet"]

    BASE_RATE_KEYWORDS = ["293", "$293"]
    EXHIBIT_1_KEYWORDS = ["exhibit 1", "exhibit i", "page 4"]

    FACTOR_KEYWORDS = ["2.061", "2.06"]
    EXHIBIT_6_KEYWORDS = ["exhibit 6", "exhibit vi", "page 71"]
    FACTOR_CRITERIA_KEYWORDS = ["ho3", "750,000", "750000", "$750,000", "2%"]

    ANSWER_KEYWORDS = ["604", "$604", "603.87", "603.873"]

    def __init__(self):
        pass

    def score_document_retrieval(
        self,
        router_selected_docs: List[str],
        retrieval_steps: List[Dict[str, Any]]
    ) -> Tuple[float, str]:
        """
        Score document retrieval (max 20 points).

        - 10 pts: Retrieved CT MAPS Homeowner Rules Manual
        - 10 pts: Retrieved CT Homeowners MAPS Rate Pages

        Args:
            router_selected_docs: List of documents selected by router
            retrieval_steps: Detailed retrieval steps with chunks

        Returns:
            (score, reasoning)
        """
        score = 0.0
        reasoning_parts = []

        # Check router selection
        router_docs_str = " ".join(router_selected_docs).lower()

        # Extract all source files from retrieval chunks
        retrieved_files = set()
        for step in retrieval_steps:
            for chunk in step.get("chunks", []):
                source = chunk.get("source_file", "")
                if source:
                    retrieved_files.add(source.lower())

        all_docs_str = (router_docs_str + " " + " ".join(retrieved_files)).lower()

        # Check for Rules Manual
        if "maps homeowner rules manual" in all_docs_str or "180449588" in all_docs_str:
            score += 10
            reasoning_parts.append("✓ Retrieved CT MAPS Homeowner Rules Manual (+10)")
        else:
            reasoning_parts.append("✗ Missing CT MAPS Homeowner Rules Manual (0)")

        # Check for Rate Pages
        if "maps rate pages" in all_docs_str or "180407973" in all_docs_str:
            score += 10
            reasoning_parts.append("✓ Retrieved CT Homeowners MAPS Rate Pages (+10)")
        else:
            reasoning_parts.append("✗ Missing CT Homeowners MAPS Rate Pages (0)")

        reasoning = " | ".join(reasoning_parts)
        return score, reasoning

    def score_deductible_identification(self, answer: str) -> Tuple[float, str]:
        """
        Score deductible identification (max 20 points).

        - 15 pts: Identifies 2% deductible
        - 5 pts: References Rule C-7 or >2,500 feet criterion

        Args:
            answer: The generated answer text

        Returns:
            (score, reasoning)
        """
        score = 0.0
        reasoning_parts = []
        answer_lower = answer.lower()

        # Check for 2% deductible
        if any(kw in answer_lower for kw in self.DEDUCTIBLE_KEYWORDS):
            score += 15
            reasoning_parts.append("✓ Identified 2% deductible (+15)")
        else:
            reasoning_parts.append("✗ Did not identify 2% deductible (0)")

        # Check for Rule C-7 or distance criterion
        if any(kw in answer_lower for kw in self.RULE_C7_KEYWORDS):
            score += 5
            reasoning_parts.append("✓ Referenced Rule C-7 or distance criterion (+5)")
        else:
            reasoning_parts.append("✗ No reference to Rule C-7 or distance (0)")

        reasoning = " | ".join(reasoning_parts)
        return score, reasoning

    def score_base_rate_identification(self, answer: str) -> Tuple[float, str]:
        """
        Score base rate identification (max 20 points).

        - 15 pts: Finds $293 hurricane base rate
        - 5 pts: References Exhibit 1 or Page 4

        Args:
            answer: The generated answer text

        Returns:
            (score, reasoning)
        """
        score = 0.0
        reasoning_parts = []
        answer_lower = answer.lower()

        # Check for $293 base rate
        if any(kw in answer_lower for kw in self.BASE_RATE_KEYWORDS):
            score += 15
            reasoning_parts.append("✓ Found $293 base rate (+15)")
        else:
            reasoning_parts.append("✗ Did not find $293 base rate (0)")

        # Check for Exhibit 1 reference
        if any(kw in answer_lower for kw in self.EXHIBIT_1_KEYWORDS):
            score += 5
            reasoning_parts.append("✓ Referenced Exhibit 1 or Page 4 (+5)")
        else:
            reasoning_parts.append("✗ No reference to Exhibit 1 (0)")

        reasoning = " | ".join(reasoning_parts)
        return score, reasoning

    def score_factor_identification(self, answer: str) -> Tuple[float, str]:
        """
        Score factor identification (max 20 points).

        - 15 pts: Finds factor 2.061
        - 5 pts: References Exhibit 6, Page 71, or HO3/$750k/2% criteria

        Args:
            answer: The generated answer text

        Returns:
            (score, reasoning)
        """
        score = 0.0
        reasoning_parts = []
        answer_lower = answer.lower()

        # Check for factor 2.061
        if any(kw in answer for kw in self.FACTOR_KEYWORDS):
            score += 15
            reasoning_parts.append("✓ Found factor 2.061 (+15)")
        else:
            reasoning_parts.append("✗ Did not find factor 2.061 (0)")

        # Check for Exhibit 6 or criteria references
        has_exhibit_ref = any(kw in answer_lower for kw in self.EXHIBIT_6_KEYWORDS)
        has_criteria_ref = any(kw in answer_lower for kw in self.FACTOR_CRITERIA_KEYWORDS)

        if has_exhibit_ref or has_criteria_ref:
            score += 5
            reasoning_parts.append("✓ Referenced Exhibit 6 or factor criteria (+5)")
        else:
            reasoning_parts.append("✗ No reference to Exhibit 6 or criteria (0)")

        reasoning = " | ".join(reasoning_parts)
        return score, reasoning

    def score_calculation(self, answer: str) -> Tuple[float, str]:
        """
        Score final calculation (max 20 points).

        - 20 pts: Correct final answer ($604 or ~603.87)
        - 10 pts: Shows correct formula even if inputs wrong

        Args:
            answer: The generated answer text

        Returns:
            (score, reasoning)
        """
        score = 0.0
        reasoning_parts = []
        answer_lower = answer.lower()

        # Check for correct final answer
        if any(kw in answer for kw in self.ANSWER_KEYWORDS):
            score += 20
            reasoning_parts.append("✓ Correct final answer $604 (+20)")
        else:
            # Check for formula indication (multiplication of base rate × factor)
            has_multiplication = ("×" in answer or "*" in answer or
                                 "multiply" in answer_lower or
                                 "multiplied" in answer_lower)
            mentions_rate = any(kw in answer_lower for kw in ["rate", "base"])
            mentions_factor = "factor" in answer_lower

            if has_multiplication and mentions_rate and mentions_factor:
                score += 10
                reasoning_parts.append("✓ Shows correct formula structure (+10)")
            else:
                reasoning_parts.append("✗ Incorrect answer and no clear formula (0)")

        reasoning = " | ".join(reasoning_parts)
        return score, reasoning

    def score_result(self, result: Dict[str, Any]) -> Tuple[ComponentScore, Dict[str, str]]:
        """
        Score a single experiment result for EF_2.

        Args:
            result: DetailedQuestionResult dict

        Returns:
            (ComponentScore, reasoning_dict)
        """
        scores = ComponentScore()
        reasoning = {}

        answer = result.get("actual_output", "")
        router_docs = result.get("router_selected_docs", [])
        retrieval_steps = result.get("retrieval_steps", [])

        # Score each component
        scores.document_retrieval, reasoning["document_retrieval"] = \
            self.score_document_retrieval(router_docs, retrieval_steps)

        scores.deductible_identification, reasoning["deductible_identification"] = \
            self.score_deductible_identification(answer)

        scores.base_rate_identification, reasoning["base_rate_identification"] = \
            self.score_base_rate_identification(answer)

        scores.factor_identification, reasoning["factor_identification"] = \
            self.score_factor_identification(answer)

        scores.calculation, reasoning["calculation"] = \
            self.score_calculation(answer)

        return scores, reasoning


def load_results(json_path: str) -> List[Dict[str, Any]]:
    """Load experiment results from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Handle both list of results and dict with "results" key
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and "results" in data:
        return data["results"]
    else:
        raise ValueError("Unknown JSON structure")


def analyze_results(json_path: str, output_csv: str = None) -> pd.DataFrame:
    """
    Analyze experiment results and generate component scores.

    Args:
        json_path: Path to detailed_results JSON file
        output_csv: Optional path to save results CSV

    Returns:
        DataFrame with scores and configuration details
    """
    results = load_results(json_path)
    scorer = EF2Scorer()

    analyzed_results = []

    for result in results:
        # Only score EF_2 questions
        if result.get("question_id") != "EF_2":
            continue

        # Score the result
        scores, reasoning = scorer.score_result(result)

        # Extract configuration for grouping
        config = result.get("config", {})
        retriever_config = config.get("retriever", {})

        analyzed_results.append({
            # Configuration
            "variation_name": result.get("variation_name", ""),
            "chunk_size": retriever_config.get("chunk_size", ""),
            "top_k": retriever_config.get("top_k_per_step", ""),
            "expand_context": retriever_config.get("expand_context", ""),
            "top_k_docs": config.get("router", {}).get("top_k_docs", ""),
            "chunking_strategy": retriever_config.get("chunking_strategy", ""),
            "use_hybrid": retriever_config.get("use_hybrid", False),
            "hybrid_alpha": retriever_config.get("hybrid_alpha", ""),

            # Component Scores
            "doc_retrieval": scores.document_retrieval,
            "deductible_id": scores.deductible_identification,
            "base_rate_id": scores.base_rate_identification,
            "factor_id": scores.factor_identification,
            "calculation": scores.calculation,
            "total_score": scores.total,

            # Reasoning
            "doc_retrieval_reason": reasoning["document_retrieval"],
            "deductible_reason": reasoning["deductible_identification"],
            "base_rate_reason": reasoning["base_rate_identification"],
            "factor_reason": reasoning["factor_identification"],
            "calculation_reason": reasoning["calculation"],

            # Metadata
            "answer_preview": result.get("actual_output", "")[:200],
            "execution_time": result.get("execution_time_seconds", 0),
            "chunks_retrieved": result.get("total_chunks_retrieved", 0),
        })

    df = pd.DataFrame(analyzed_results)

    # Sort by total score descending
    df = df.sort_values("total_score", ascending=False)

    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"✓ Results saved to {output_csv}")

    return df


def print_top_configurations(df: pd.DataFrame, top_n: int = 10):
    """Print summary of top-performing configurations."""
    print("\n" + "="*80)
    print(f"TOP {top_n} CONFIGURATIONS FOR EF_2 (by Component Score)")
    print("="*80)

    # Display columns
    display_cols = [
        "variation_name", "chunk_size", "top_k", "expand_context",
        "top_k_docs", "chunking_strategy",
        "doc_retrieval", "deductible_id", "base_rate_id",
        "factor_id", "calculation", "total_score"
    ]

    top_df = df[display_cols].head(top_n)

    print(top_df.to_string(index=False))
    print("\n")

    # Summary statistics
    print("="*80)
    print("COMPONENT STATISTICS")
    print("="*80)

    component_cols = ["doc_retrieval", "deductible_id", "base_rate_id", "factor_id", "calculation"]
    stats = df[component_cols].describe()
    print(stats)
    print("\n")

    # Best per component
    print("="*80)
    print("BEST CONFIGURATIONS PER COMPONENT")
    print("="*80)

    for component in component_cols:
        best_idx = df[component].idxmax()
        best_row = df.loc[best_idx]
        print(f"\n{component.upper()}:")
        print(f"  Score: {best_row[component]:.1f}/20")
        print(f"  Config: {best_row['variation_name']}")
        print(f"  Reason: {best_row[component + '_reason']}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Score EF_2 experiment results with component-based evaluation"
    )
    parser.add_argument(
        "json_file",
        help="Path to detailed_results JSON file"
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output CSV path (default: ef2_component_scores.csv)",
        default="ef2_component_scores.csv"
    )
    parser.add_argument(
        "--top",
        "-t",
        type=int,
        default=10,
        help="Number of top configurations to display (default: 10)"
    )

    args = parser.parse_args()

    # Validate input file
    if not Path(args.json_file).exists():
        print(f"❌ Error: File not found: {args.json_file}")
        return 1

    print(f"Loading results from {args.json_file}...")
    df = analyze_results(args.json_file, args.output)

    print(f"\n✓ Analyzed {len(df)} EF_2 results")

    # Print summary
    print_top_configurations(df, args.top)

    print(f"\n✓ Full results saved to {args.output}")
    print("\nTo explore further, load the CSV in pandas or Excel:")
    print(f"  df = pd.read_csv('{args.output}')")

    return 0


if __name__ == "__main__":
    exit(main())

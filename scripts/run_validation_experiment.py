#!/usr/bin/env python3
"""
Run validation experiments to compare enhanced prompts vs baseline.

This script runs a small set of experiments using the top configurations
from the baseline experiments to validate the enhanced prompt improvements.

Usage:
    python scripts/run_validation_experiment.py [--question EF_1|EF_2|both]

Examples:
    python scripts/run_validation_experiment.py                    # Run both questions
    python scripts/run_validation_experiment.py --question EF_2    # Run only EF_2
"""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from answer_pdf_question import answer_pdf_question
from src.agents import (
    OrchestratorConfig, RouterConfig, PlannerConfig, RetrieverConfig,
    OrchestratorAgent
)


# Top configurations from baseline experiments
TOP_CONFIGS = {
    "EF_1": {
        "name": "chunk_2000_topk_5_window0_docs1",
        "description": "Best EF_1 config: chunk=2000, top_k=5, expand=0, docs=1",
        "router": {"top_k_docs": 1},
        "retriever": {
            "top_k_per_step": 5,
            "chunk_size": 2000,
            "chunk_overlap": 200,
            "expand_context": 0,
            "chunking_strategy": "page",
            "use_hybrid": True,
            "hybrid_alpha": 0.5
        }
    },
    "EF_2": {
        "name": "chunk_400_topk_3_window0_docs5",
        "description": "Best EF_2 config: chunk=400, top_k=3, expand=0, docs=5",
        "router": {"top_k_docs": 5},
        "retriever": {
            "top_k_per_step": 3,
            "chunk_size": 400,
            "chunk_overlap": 200,
            "expand_context": 0,
            "chunking_strategy": "document",
            "use_hybrid": True,
            "hybrid_alpha": 0.5
        }
    }
}

QUESTIONS = {
    "EF_1": {
        "question": "List all rating plan rules",
        "expected": "C-1, C-2, C-3, C-4, C-5, C-6, C-7, C-8, C-9, C-10, C-11, C-12, C-13, C-14, C-15, C-16, C-17, C-18, C-19, C-20, C-21, C-22, C-23, C-24, C-25, C-26, C-27, C-28, C-29, C-30, C-31, C-32, C-33, C-34, C-35",
        "pdfs_folder": "artifacts/1"
    },
    "EF_2": {
        "question": "Using the Homeowner MAPS Rate Pages and the Rules Manual, calculate the unadjusted Hurricane premium for an HO3 policy with a $750,000 Coverage A limit. The property is located 3,000 feet from the coast in a Coastline Neighborhood.",
        "expected": "$604",
        "pdfs_folder": "artifacts/1"
    }
}


def run_single_experiment(question_id: str, config: dict) -> dict:
    """Run a single experiment and return results."""
    q = QUESTIONS[question_id]

    print(f"\n{'='*60}")
    print(f"Running {question_id} with config: {config['name']}")
    print(f"{'='*60}")
    print(f"Question: {q['question'][:80]}...")

    # Create configs
    router_config = RouterConfig(**config['router'])
    retriever_config = RetrieverConfig(**config['retriever'])

    orchestrator_config = OrchestratorConfig(
        router_config=router_config,
        retriever_config=retriever_config
    )

    # Run experiment
    start_time = time.time()
    try:
        orchestrator = OrchestratorAgent(config=orchestrator_config)
        result = orchestrator.answer_question(
            question=q['question'],
            pdfs_folder=q['pdfs_folder'],
            summaries_path="artifacts/document_summaries.json",
            verbose=True
        )
        answer = result.answer
        error = None
    except Exception as e:
        answer = f"ERROR: {str(e)}"
        error = str(e)

    execution_time = time.time() - start_time

    print(f"\n{'─'*60}")
    print(f"ANSWER ({len(answer)} chars):")
    print(f"{'─'*60}")
    print(answer[:500] + "..." if len(answer) > 500 else answer)
    print(f"\nExecution time: {execution_time:.1f}s")

    return {
        "question_id": question_id,
        "question": q['question'],
        "expected": q['expected'],
        "answer": answer,
        "config_name": config['name'],
        "config": config,
        "execution_time": execution_time,
        "error": error
    }


def score_ef2_result(result: dict) -> dict:
    """Score EF_2 result using component scoring."""
    answer = result['answer'].lower()

    scores = {
        "document_retrieval": 0,
        "deductible_id": 0,
        "base_rate_id": 0,
        "factor_id": 0,
        "calculation": 0
    }

    # Document retrieval (simplified - check if answer exists)
    if result['error'] is None and len(result['answer']) > 100:
        scores["document_retrieval"] = 20

    # Deductible identification
    if "2%" in answer or "2 percent" in answer:
        scores["deductible_id"] += 15
    if "c-7" in answer or "rule c-7" in answer or "2500" in answer or "2,500" in answer:
        scores["deductible_id"] += 5

    # Base rate identification
    if "293" in answer:
        scores["base_rate_id"] += 15
    if "exhibit 1" in answer or "page 4" in answer:
        scores["base_rate_id"] += 5

    # Factor identification
    if "2.061" in answer or "2.06" in answer:
        scores["factor_id"] += 15
    if "exhibit 6" in answer or "page 71" in answer:
        scores["factor_id"] += 5

    # Calculation
    if "604" in answer:
        scores["calculation"] = 20
    elif "×" in answer or "*" in answer or "multiply" in answer:
        scores["calculation"] = 10

    scores["total"] = sum(scores.values())

    return scores


def score_ef1_result(result: dict) -> dict:
    """Score EF_1 result using partial match."""
    expected_rules = ["C-1", "C-2", "C-3", "C-4", "C-5", "C-6", "C-7", "C-8", "C-9", "C-10",
                      "C-11", "C-12", "C-13", "C-14", "C-15", "C-16", "C-17", "C-18", "C-19", "C-20",
                      "C-21", "C-22", "C-23", "C-24", "C-25", "C-26", "C-27", "C-28", "C-29", "C-30",
                      "C-31", "C-32", "C-33", "C-34", "C-35"]

    found = 0
    for rule in expected_rules:
        if rule.lower() in result['answer'].lower():
            found += 1

    return {
        "rules_found": found,
        "total_rules": len(expected_rules),
        "partial_match_score": found / len(expected_rules)
    }


def main():
    parser = argparse.ArgumentParser(description="Run validation experiments")
    parser.add_argument("--question", choices=["EF_1", "EF_2", "both"], default="both",
                        help="Which question to test")
    args = parser.parse_args()

    print("\n" + "="*70)
    print("VALIDATION EXPERIMENT - Enhanced Prompts")
    print("="*70)
    print(f"Date: {datetime.now().isoformat()}")
    print(f"Testing: {args.question}")
    print("="*70)

    results = []

    if args.question in ["EF_1", "both"]:
        result = run_single_experiment("EF_1", TOP_CONFIGS["EF_1"])
        scores = score_ef1_result(result)
        result["scores"] = scores
        results.append(result)

        print(f"\nEF_1 Score: {scores['rules_found']}/{scores['total_rules']} rules found ({scores['partial_match_score']:.1%})")

    if args.question in ["EF_2", "both"]:
        result = run_single_experiment("EF_2", TOP_CONFIGS["EF_2"])
        scores = score_ef2_result(result)
        result["scores"] = scores
        results.append(result)

        print(f"\nEF_2 Component Scores:")
        print(f"  Document Retrieval: {scores['document_retrieval']}/20")
        print(f"  Deductible ID:      {scores['deductible_id']}/20")
        print(f"  Base Rate ID:       {scores['base_rate_id']}/20")
        print(f"  Factor ID:          {scores['factor_id']}/20")
        print(f"  Calculation:        {scores['calculation']}/20")
        print(f"  TOTAL:              {scores['total']}/100")

    # Save results
    output_path = f"results/validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    Path("results").mkdir(exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "description": "Validation experiment with enhanced prompts",
            "results": results
        }, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print("VALIDATION COMPLETE")
    print(f"{'='*70}")
    print(f"Results saved to: {output_path}")

    # Print comparison guidance
    print("\n" + "─"*70)
    print("COMPARISON WITH BASELINE")
    print("─"*70)
    print("Baseline results (from experiment_summary.json):")
    print("  EF_1: 75.8% partial match")
    print("  EF_2: 55/100 (20+20+0+5+10)")
    print("\nExpected improvements with enhanced prompts:")
    print("  EF_2 Base Rate ID: 0 → >0 (targeting Exhibit 1)")
    print("  EF_2 Factor ID: 5 → >5 (targeting Exhibit 6)")
    print("  EF_2 Calculation: 10 → 20 (if values extracted)")


if __name__ == "__main__":
    main()

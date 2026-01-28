#!/usr/bin/env python3
"""
Simple validation script to test 5 scenarios with enhanced prompts.

Runs 5 configurations and compares against baseline results.

Usage:
    python scripts/test_5_scenarios.py
"""
import json
import time
from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from answer_pdf_question import answer_pdf_question
from src.agents import OrchestratorConfig, RouterConfig, RetrieverConfig

# 5 Test Scenarios: Mix of EF_1 and EF_2 top configs
SCENARIOS = [
    {
        "id": "S1_EF2_best",
        "question_id": "EF_2",
        "config": {"chunk_size": 400, "top_k": 3, "expand": 0, "docs": 5, "strategy": "document"},
        "baseline_score": 55
    },
    {
        "id": "S2_EF2_large_chunk",
        "question_id": "EF_2",
        "config": {"chunk_size": 2000, "top_k": 1, "expand": 0, "docs": 5, "strategy": "page"},
        "baseline_score": 55
    },
    {
        "id": "S3_EF2_medium",
        "question_id": "EF_2",
        "config": {"chunk_size": 800, "top_k": 3, "expand": 0, "docs": 5, "strategy": "page"},
        "baseline_score": 55
    },
    {
        "id": "S4_EF1_best",
        "question_id": "EF_1",
        "config": {"chunk_size": 2000, "top_k": 5, "expand": 0, "docs": 1, "strategy": "page"},
        "baseline_score": 75.8
    },
    {
        "id": "S5_EF1_window",
        "question_id": "EF_1",
        "config": {"chunk_size": 1200, "top_k": 5, "expand": 2, "docs": 1, "strategy": "page"},
        "baseline_score": 69.7
    }
]

QUESTIONS = {
    "EF_1": "List all rating plan rules",
    "EF_2": "Using the Homeowner MAPS Rate Pages and the Rules Manual, calculate the unadjusted Hurricane premium for an HO3 policy with a $750,000 Coverage A limit. The property is located 3,000 feet from the coast in a Coastline Neighborhood."
}


def score_ef2(answer: str) -> dict:
    """Score EF_2 answer using component scoring."""
    answer_lower = answer.lower()
    scores = {"doc": 0, "deduct": 0, "base": 0, "factor": 0, "calc": 0}

    # Document retrieval (assume success if answer exists)
    if len(answer) > 100:
        scores["doc"] = 20

    # Deductible (2% + Rule C-7)
    if "2%" in answer_lower or "2 percent" in answer_lower:
        scores["deduct"] += 15
    if "c-7" in answer_lower or "2500" in answer_lower or "2,500" in answer_lower:
        scores["deduct"] += 5

    # Base rate ($293 + Exhibit 1)
    if "293" in answer:
        scores["base"] += 15
    if "exhibit 1" in answer_lower or "page 4" in answer_lower:
        scores["base"] += 5

    # Factor (2.061 + Exhibit 6)
    if "2.061" in answer or "2.06" in answer:
        scores["factor"] += 15
    if "exhibit 6" in answer_lower or "page 71" in answer_lower:
        scores["factor"] += 5

    # Calculation ($604 or formula)
    if "604" in answer:
        scores["calc"] = 20
    elif "×" in answer or "*" in answer or "multiply" in answer_lower:
        scores["calc"] = 10

    scores["total"] = sum(scores.values())
    return scores


def score_ef1(answer: str) -> dict:
    """Score EF_1 answer by counting rules found."""
    rules = [f"C-{i}" for i in range(1, 36)]
    found = sum(1 for r in rules if r.lower() in answer.lower())
    return {"rules_found": found, "total_rules": 35, "percent": round(found / 35 * 100, 1)}


def run_scenario(scenario: dict) -> dict:
    """Run a single test scenario."""
    cfg = scenario["config"]
    q = QUESTIONS[scenario["question_id"]]

    print(f"\n{'─'*60}")
    print(f"Running: {scenario['id']}")
    print(f"Config: chunk={cfg['chunk_size']}, top_k={cfg['top_k']}, docs={cfg['docs']}")
    print(f"{'─'*60}")

    # Build config
    orchestrator_config = OrchestratorConfig(
        router_config=RouterConfig(top_k_docs=cfg["docs"]),
        retriever_config=RetrieverConfig(
            top_k_per_step=cfg["top_k"],
            chunk_size=cfg["chunk_size"],
            expand_context=cfg["expand"],
            chunking_strategy=cfg["strategy"],
            use_hybrid=True,
            hybrid_alpha=0.5
        )
    )

    start = time.time()
    try:
        from src.agents import OrchestratorAgent
        # OrchestratorAgent requires summaries_path and pdf_folder in __init__
        agent = OrchestratorAgent(
            summaries_path="artifacts/document_summaries.json",
            pdf_folder="artifacts/1",
            config=orchestrator_config
        )
        result = agent.answer_question(
            question=q,
            verbose=True
        )
        answer = result.answer
        error = None
    except Exception as e:
        answer = f"ERROR: {e}"
        error = str(e)
        import traceback
        traceback.print_exc()

    elapsed = time.time() - start

    # Score based on question type
    if scenario["question_id"] == "EF_2":
        scores = score_ef2(answer)
        score_display = f"{scores['total']}/100"
        improvement = scores['total'] - scenario['baseline_score']
    else:
        scores = score_ef1(answer)
        score_display = f"{scores['percent']}%"
        improvement = scores['percent'] - scenario['baseline_score']

    print(f"Answer length: {len(answer)} chars")
    print(f"Score: {score_display} (baseline: {scenario['baseline_score']}, diff: {improvement:+.1f})")
    print(f"Time: {elapsed:.1f}s")

    return {
        "scenario_id": scenario["id"],
        "question_id": scenario["question_id"],
        "config": cfg,
        "answer": answer[:500] + "..." if len(answer) > 500 else answer,
        "scores": scores,
        "baseline_score": scenario["baseline_score"],
        "improvement": improvement,
        "execution_time": elapsed,
        "error": error
    }


def main():
    print("="*70)
    print("5 SCENARIO VALIDATION TEST - Enhanced Prompts")
    print("="*70)
    print(f"Started: {datetime.now().isoformat()}")

    results = []
    for scenario in SCENARIOS:
        result = run_scenario(scenario)
        results.append(result)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY: Enhanced Prompts vs Baseline")
    print("="*70)
    print(f"\n{'Scenario':<20} {'Type':<6} {'New Score':<12} {'Baseline':<10} {'Change':<10}")
    print("-"*60)

    ef2_improvements = []
    ef1_improvements = []

    for r in results:
        if r["question_id"] == "EF_2":
            score_str = f"{r['scores']['total']}/100"
            baseline_str = f"{r['baseline_score']}/100"
            ef2_improvements.append(r["improvement"])
        else:
            score_str = f"{r['scores']['percent']}%"
            baseline_str = f"{r['baseline_score']}%"
            ef1_improvements.append(r["improvement"])

        change = f"{r['improvement']:+.1f}"
        print(f"{r['scenario_id']:<20} {r['question_id']:<6} {score_str:<12} {baseline_str:<10} {change:<10}")

    print("-"*60)
    if ef2_improvements:
        avg_ef2 = sum(ef2_improvements) / len(ef2_improvements)
        print(f"EF_2 Average Change: {avg_ef2:+.1f} points")
    if ef1_improvements:
        avg_ef1 = sum(ef1_improvements) / len(ef1_improvements)
        print(f"EF_1 Average Change: {avg_ef1:+.1f}%")

    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "description": "5 scenario validation with enhanced prompts",
        "results": results,
        "summary": {
            "ef2_avg_improvement": sum(ef2_improvements) / len(ef2_improvements) if ef2_improvements else 0,
            "ef1_avg_improvement": sum(ef1_improvements) / len(ef1_improvements) if ef1_improvements else 0
        }
    }

    output_path = f"results/validation_5scenarios_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    Path("results").mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved: {output_path}")
    print("\nTo compare component scores for EF_2:")
    print("  Baseline: doc=20, deduct=20, base=0, factor=5, calc=10")
    print("  Look for improvements in 'base' and 'factor' components")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
A/B Testing Script for EF1 & EF2 with Decoding Parameter Variations.

Tests top 5 configurations from each question with creative vs high-precision
decoding parameters using existing scoring approaches.

Usage:
    python scripts/run_ab_test.py [--question EF_1|EF_2|both] [--track creative|precision|both]

Examples:
    python scripts/run_ab_test.py                          # Run all tests
    python scripts/run_ab_test.py --question EF_2          # Only EF_2
    python scripts/run_ab_test.py --track precision        # Only high-precision track
"""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents import OrchestratorConfig, RouterConfig, RetrieverConfig, OrchestratorAgent

# Decoding parameter variants (A/B for each track)
DECODING_VARIANTS = {
    "creative": {
        "A": {"temperature": 0.7, "top_k": 40, "top_p": 0.9, "repeat_penalty": 1.1, "seed": 42},
        "B": {"temperature": 0.9, "top_k": 80, "top_p": 0.95, "repeat_penalty": 1.0, "seed": 43}
    },
    "precision": {
        "A": {"temperature": 0.0, "top_k": 1, "top_p": 1.0, "repeat_penalty": 1.2, "seed": 123},
        "B": {"temperature": 0.2, "top_k": 10, "top_p": 0.9, "repeat_penalty": 1.2, "seed": 124}
    }
}

# Top 5 configurations from experiment_summary.json
TOP_5_CONFIGS = {
    "EF_1": [
        {"name": "chunk_2000_topk_5_window0_docs1_page", "chunk_size": 2000, "top_k": 5, "expand": 0, "docs": 1, "strategy": "page"},
        {"name": "chunk_2000_topk_5_window0_docs5_page", "chunk_size": 2000, "top_k": 5, "expand": 0, "docs": 5, "strategy": "page"},
        {"name": "chunk_2000_topk_5_window0_docs1_document", "chunk_size": 2000, "top_k": 5, "expand": 0, "docs": 1, "strategy": "document"},
        {"name": "chunk_2000_topk_5_window0_docs5_document", "chunk_size": 2000, "top_k": 5, "expand": 0, "docs": 5, "strategy": "document"},
        {"name": "chunk_1200_topk_5_window2_docs1_page", "chunk_size": 1200, "top_k": 5, "expand": 2, "docs": 1, "strategy": "page"},
    ],
    "EF_2": [
        {"name": "chunk_400_topk_3_window0_docs5_document", "chunk_size": 400, "top_k": 3, "expand": 0, "docs": 5, "strategy": "document"},
        {"name": "chunk_2000_topk_1_window0_docs5_page", "chunk_size": 2000, "top_k": 1, "expand": 0, "docs": 5, "strategy": "page"},
        {"name": "chunk_800_topk_1_window1_docs5_document", "chunk_size": 800, "top_k": 1, "expand": 1, "docs": 5, "strategy": "document"},
        {"name": "chunk_400_topk_5_window0_docs5_page", "chunk_size": 400, "top_k": 5, "expand": 0, "docs": 5, "strategy": "page"},
        {"name": "chunk_400_topk_3_window0_docs5_page", "chunk_size": 400, "top_k": 3, "expand": 0, "docs": 5, "strategy": "page"},
    ]
}

QUESTIONS = {
    "EF_1": "List all rating plan rules",
    "EF_2": "Using the Homeowner MAPS Rate Pages and the Rules Manual, calculate the unadjusted Hurricane premium for an HO3 policy with a $750,000 Coverage A limit. The property is located 3,000 feet from the coast in a Coastline Neighborhood."
}

EXPECTED_RULES = [f"C-{i}" for i in range(1, 36)]


def score_ef1(answer: str) -> dict:
    """Score EF_1 using partial match (coverage of expected rules)."""
    answer_lower = answer.lower()
    found = sum(1 for rule in EXPECTED_RULES if rule.lower() in answer_lower)
    coverage = found / len(EXPECTED_RULES)
    return {
        "rules_found": found,
        "total_rules": len(EXPECTED_RULES),
        "coverage": round(coverage * 100, 1),
        "score": round(coverage * 100, 1)
    }


def score_ef2(answer: str) -> dict:
    """Score EF_2 using component-based scoring (100 points total)."""
    answer_lower = answer.lower()
    scores = {
        "document_retrieval": 0,
        "deductible_id": 0,
        "base_rate_id": 0,
        "factor_id": 0,
        "calculation": 0
    }

    # Document retrieval (assume success if answer exists)
    if len(answer) > 100:
        scores["document_retrieval"] = 20

    # Deductible (2% + Rule C-7)
    if "2%" in answer_lower or "2 percent" in answer_lower:
        scores["deductible_id"] += 15
    if "c-7" in answer_lower or "2500" in answer_lower or "2,500" in answer_lower:
        scores["deductible_id"] += 5

    # Base rate ($293 + Exhibit 1)
    if "293" in answer:
        scores["base_rate_id"] += 15
    if "exhibit 1" in answer_lower or "page 4" in answer_lower:
        scores["base_rate_id"] += 5

    # Factor (2.061 + Exhibit 6)
    if "2.061" in answer or "2.06" in answer:
        scores["factor_id"] += 15
    if "exhibit 6" in answer_lower or "page 71" in answer_lower:
        scores["factor_id"] += 5

    # Calculation ($604 or formula)
    if "604" in answer:
        scores["calculation"] = 20
    elif "×" in answer or "*" in answer or "multiply" in answer_lower:
        scores["calculation"] = 10

    scores["total"] = sum(scores.values())
    scores["score"] = scores["total"]
    return scores


def run_single_test(question_id: str, config: dict, decoding: dict, track: str, variant: str) -> dict:
    """Run a single A/B test."""
    q = QUESTIONS[question_id]

    # Build orchestrator config
    orchestrator_config = OrchestratorConfig(
        temperature=decoding.get("temperature", 0.0),
        router_config=RouterConfig(top_k_docs=config["docs"]),
        retriever_config=RetrieverConfig(
            top_k_per_step=config["top_k"],
            chunk_size=config["chunk_size"],
            expand_context=config["expand"],
            chunking_strategy=config["strategy"],
            use_hybrid=True,
            hybrid_alpha=0.5
        )
    )

    start = time.time()
    try:
        agent = OrchestratorAgent(
            summaries_path="artifacts/document_summaries.json",
            pdf_folder="artifacts/1",
            config=orchestrator_config
        )
        result = agent.answer_question(question=q, verbose=False)
        answer = result.answer
        error = None
    except Exception as e:
        answer = f"ERROR: {e}"
        error = str(e)

    elapsed = time.time() - start

    # Score
    if question_id == "EF_1":
        scores = score_ef1(answer)
    else:
        scores = score_ef2(answer)

    return {
        "run_id": f"{question_id.lower()}-{track}-{variant}-{config['name'][:20]}",
        "question_id": question_id,
        "track": track,
        "variant": variant,
        "config_name": config["name"],
        "retriever_config": config,
        "decoding_params": decoding,
        "answer": answer[:500] + "..." if len(answer) > 500 else answer,
        "scores": scores,
        "execution_time": elapsed,
        "error": error,
        "timestamp": datetime.now().isoformat()
    }


def run_ab_tests(question_ids: list, tracks: list) -> list:
    """Run all A/B tests for specified questions and tracks."""
    results = []
    total_tests = sum(
        len(TOP_5_CONFIGS[q]) * len(tracks) * 2  # 2 variants per track
        for q in question_ids
    )

    print(f"\n{'='*70}")
    print(f"A/B TESTING: {len(question_ids)} questions × {len(tracks)} tracks × 5 configs × 2 variants")
    print(f"Total tests: {total_tests}")
    print(f"{'='*70}")

    test_num = 0
    for question_id in question_ids:
        configs = TOP_5_CONFIGS[question_id]

        for track in tracks:
            for variant in ["A", "B"]:
                decoding = DECODING_VARIANTS[track][variant]

                for config in configs:
                    test_num += 1
                    print(f"\n[{test_num}/{total_tests}] {question_id} | {track}-{variant} | {config['name'][:30]}")

                    result = run_single_test(question_id, config, decoding, track, variant)
                    results.append(result)

                    print(f"  Score: {result['scores']['score']} | Time: {result['execution_time']:.1f}s")

    return results


def summarize_results(results: list) -> dict:
    """Generate summary comparing A vs B for each track/question."""
    summary = {"by_question": {}, "by_track": {}}

    for question_id in ["EF_1", "EF_2"]:
        q_results = [r for r in results if r["question_id"] == question_id]
        if not q_results:
            continue

        summary["by_question"][question_id] = {}

        for track in ["creative", "precision"]:
            track_results = [r for r in q_results if r["track"] == track]
            if not track_results:
                continue

            a_results = [r for r in track_results if r["variant"] == "A"]
            b_results = [r for r in track_results if r["variant"] == "B"]

            a_avg = sum(r["scores"]["score"] for r in a_results) / len(a_results) if a_results else 0
            b_avg = sum(r["scores"]["score"] for r in b_results) / len(b_results) if b_results else 0

            summary["by_question"][question_id][track] = {
                "variant_A_avg": round(a_avg, 1),
                "variant_B_avg": round(b_avg, 1),
                "delta": round(b_avg - a_avg, 1),
                "winner": "B" if b_avg > a_avg else "A" if a_avg > b_avg else "tie"
            }

    return summary


def print_summary(summary: dict, results: list):
    """Print formatted summary."""
    print(f"\n{'='*70}")
    print("A/B TEST RESULTS SUMMARY")
    print(f"{'='*70}")

    print(f"\n{'Question':<10} {'Track':<12} {'Variant A':<12} {'Variant B':<12} {'Delta':<10} {'Winner':<8}")
    print("-" * 70)

    for question_id, tracks in summary["by_question"].items():
        for track, data in tracks.items():
            unit = "%" if question_id == "EF_1" else "/100"
            print(f"{question_id:<10} {track:<12} {data['variant_A_avg']}{unit:<6} {data['variant_B_avg']}{unit:<6} {data['delta']:+.1f}{'':<5} {data['winner']:<8}")

    print(f"\n{'='*70}")
    print("DECODING PARAMETERS")
    print(f"{'='*70}")
    for track, variants in DECODING_VARIANTS.items():
        print(f"\n{track.upper()} TRACK:")
        for var, params in variants.items():
            print(f"  {var}: temp={params['temperature']}, top_k={params['top_k']}, top_p={params['top_p']}")


def main():
    parser = argparse.ArgumentParser(description="A/B Testing for EF1 & EF2")
    parser.add_argument("--question", choices=["EF_1", "EF_2", "both"], default="both")
    parser.add_argument("--track", choices=["creative", "precision", "both"], default="both")
    args = parser.parse_args()

    # Determine questions and tracks
    question_ids = ["EF_1", "EF_2"] if args.question == "both" else [args.question]
    tracks = ["creative", "precision"] if args.track == "both" else [args.track]

    print(f"\nA/B Testing Framework - {datetime.now().isoformat()}")
    print(f"Questions: {question_ids}")
    print(f"Tracks: {tracks}")

    # Run tests
    results = run_ab_tests(question_ids, tracks)

    # Generate summary
    summary = summarize_results(results)
    print_summary(summary, results)

    # Save results
    output = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "questions": question_ids,
            "tracks": tracks,
            "total_tests": len(results)
        },
        "decoding_variants": DECODING_VARIANTS,
        "summary": summary,
        "results": results
    }

    output_path = f"results/ab_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    Path("results").mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved: {output_path}")

    # Print next steps
    print(f"\n{'='*70}")
    print("FUTURE ENHANCEMENTS (Not Implemented)")
    print(f"{'='*70}")
    print("- Beam search emulation via multi-sample voting")
    print("- LLM-as-judge for qualitative evaluation")
    print("- Additional metrics: fluency, diversity, hallucination rate")
    print("- Cross-encoder re-ranking for improved retrieval")


if __name__ == "__main__":
    main()

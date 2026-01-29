#!/usr/bin/env python3
"""
Test script for Planner Agent.

Tests multi-step plan creation for complex questions.
Demonstrates Router → Planner integration.
"""
import argparse
import json
from src.agents.router_agent import RouterAgent, RouterConfig
from src.agents.planner_agent import PlannerAgent, PlannerConfig


def test_planner():
    """Test planner agent with Router integration."""

    print("Initializing agents...")

    # Initialize Router
    router = RouterAgent(
        summaries_path="artifacts/document_summaries.json",
        config=RouterConfig(model="llama3.2", top_k_docs=3)
    )

    # Initialize Planner
    planner = PlannerAgent(
        config=PlannerConfig(model="llama3.2", temperature=0.0)
    )

    # Test cases
    test_cases = [
        {
            "id": "EF_1",
            "question": "What are the rules for an ineligible risk?",
            "expected_steps": 1,  # Simple lookup
            "expected_combination": False
        },
        {
            "id": "EF_2",
            "question": "What is the premium for a Tier 1 building with Protection Class 5, Coverage A of $500,000 and 2% deductible?",
            "expected_steps": 3,  # Multi-step: base rate, deductible factor, calculate
            "expected_combination": True
        },
        {
            "id": "CUSTOM_1",
            "question": "What are the coastal property restrictions and how do they affect rates?",
            "expected_steps": 2,  # Rules + rates
            "expected_combination": True
        }
    ]

    print(f"\n{'=' * 70}")
    print("PLANNER AGENT TEST SUITE")
    print(f"{'=' * 70}\n")

    results = []

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'═' * 70}")
        print(f"Test {i}/{len(test_cases)}: {test_case['id']}")
        print(f"{'═' * 70}")

        question = test_case['question']

        # Step 1: Router selects documents
        print(f"\n[ROUTER] Selecting documents...")
        selected_docs = router.select_documents(
            question=question,
            top_k=3,
            verbose=False
        )
        print(f"  Selected: {', '.join(selected_docs)}")

        # Step 2: Planner creates retrieval strategy
        print(f"\n[PLANNER] Creating retrieval plan...")
        plan = planner.create_plan(
            question=question,
            selected_documents=selected_docs,
            summaries=router.summaries,
            verbose=True
        )

        # Evaluate
        step_count_ok = len(plan.steps) >= test_case['expected_steps']
        combination_ok = plan.requires_combination == test_case['expected_combination']

        result = {
            "test_id": test_case['id'],
            "question": question,
            "selected_docs": selected_docs,
            "plan_steps": len(plan.steps),
            "expected_steps": test_case['expected_steps'],
            "requires_combination": plan.requires_combination,
            "expected_combination": test_case['expected_combination'],
            "step_count_ok": step_count_ok,
            "combination_ok": combination_ok,
            "strategy": plan.strategy
        }
        results.append(result)

        # Print evaluation
        print(f"\n{'─' * 70}")
        print("EVALUATION:")
        print(f"{'─' * 70}")
        print(f"  Steps: {len(plan.steps)} (expected ≥{test_case['expected_steps']}) {'✓' if step_count_ok else '✗'}")
        print(f"  Combination: {plan.requires_combination} (expected {test_case['expected_combination']}) {'✓' if combination_ok else '✗'}")
        print(f"  Strategy: {plan.strategy}")

        # Show plan details
        print(f"\nPLAN DETAILS:")
        for step in plan.steps:
            print(f"\n  Step {step.step_number}: {step.description}")
            print(f"    Documents: {', '.join(step.target_documents)}")
            print(f"    Query: \"{step.query}\"")
            print(f"    Expected: {step.expected_output}")

    # Summary
    print(f"\n{'═' * 70}")
    print("TEST SUMMARY")
    print(f"{'═' * 70}")

    passed = sum(1 for r in results if r['step_count_ok'] and r['combination_ok'])
    total = len(results)

    print(f"\nPassed: {passed}/{total} ({100 * passed / total:.0f}%)")

    print("\nDetailed Results:")
    for r in results:
        status = "✓" if (r['step_count_ok'] and r['combination_ok']) else "✗"
        print(f"\n  {status} {r['test_id']}")
        print(f"     Q: {r['question'][:65]}...")
        print(f"     Docs: {', '.join([d[:30] for d in r['selected_docs']])}")
        print(f"     Plan: {r['plan_steps']} steps, combination={r['requires_combination']}")
        print(f"     Strategy: {r['strategy'][:80]}...")

    print()


def interactive_mode():
    """Interactive mode for testing Router → Planner pipeline."""

    print("Initializing agents...")

    router = RouterAgent(
        summaries_path="artifacts/document_summaries.json",
        config=RouterConfig(model="llama3.2", top_k_docs=3)
    )

    planner = PlannerAgent(
        config=PlannerConfig(model="llama3.2", temperature=0.0)
    )

    print(f"\n{'=' * 70}")
    print("ROUTER → PLANNER INTERACTIVE MODE")
    print(f"{'=' * 70}")
    print("Type your questions to see document selection and retrieval plans.")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            question = input("Question: ").strip()

            if question.lower() in ['quit', 'exit', 'q']:
                break

            if not question:
                continue

            print(f"\n{'─' * 70}")
            print("[ROUTER] Selecting documents...")
            print(f"{'─' * 70}")

            # Route
            selected = router.select_documents(question=question, top_k=3, verbose=False)
            print(f"Selected {len(selected)} documents:")
            for doc in selected:
                doc_type = router.summaries.get(doc, {}).get('document_type', 'unknown')
                print(f"  - {doc} ({doc_type})")

            print(f"\n{'─' * 70}")
            print("[PLANNER] Creating retrieval plan...")
            print(f"{'─' * 70}")

            # Plan
            plan = planner.create_plan(
                question=question,
                selected_documents=selected,
                summaries=router.summaries,
                verbose=False
            )

            print(f"\nStrategy: {plan.strategy}")
            print(f"Steps: {len(plan.steps)}")
            print(f"Requires combination: {plan.requires_combination}")

            for step in plan.steps:
                print(f"\n  Step {step.step_number}: {step.description}")
                print(f"    Query: \"{step.query}\"")
                print(f"    Docs: {', '.join(step.target_documents)}")

            print(f"\nSuccess criteria: {plan.success_criteria}")
            print()

        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}\n")


def test_plan_refinement():
    """Test conversational plan refinement."""

    print("Initializing Planner...")
    planner = PlannerAgent(config=PlannerConfig(model="llama3.2"))

    # Load summaries
    import json
    with open("artifacts/document_summaries.json", 'r') as f:
        summaries = json.load(f)

    print(f"\n{'=' * 70}")
    print("PLAN REFINEMENT TEST")
    print(f"{'=' * 70}\n")

    question = "What is the premium for Tier 1, PC 5, $500k, 2% deductible?"
    selected_docs = [
        "(214933333-180358021)-CT Homeowners MAPS Tier Rate Pages Eff 8.18.25.pdf",
        "(215004905-180407973)-CT Homeowners MAPS Rate Pages Eff 8.18.25 v3.pdf"
    ]

    # Create initial plan
    print("[1] Creating initial plan...")
    plan = planner.create_plan(
        question=question,
        selected_documents=selected_docs,
        summaries=summaries,
        verbose=True
    )

    # Refine with feedback
    print(f"\n{'─' * 70}")
    print("[2] Refining plan with feedback...")
    print(f"{'─' * 70}")

    feedback = "Step 2 should look for deductible factors, not just the deductible value"

    refined_plan = planner.refine_plan(
        plan=plan,
        feedback=feedback,
        verbose=True
    )

    print(f"\nRefined plan:")
    for step in refined_plan.steps:
        print(f"  {step.step_number}. {step.description}")

    print()


def main():
    parser = argparse.ArgumentParser(description="Test Planner Agent")
    parser.add_argument(
        '--mode',
        choices=['test', 'interactive', 'refinement'],
        default='test',
        help='Test mode'
    )

    args = parser.parse_args()

    if args.mode == 'test':
        test_planner()
    elif args.mode == 'interactive':
        interactive_mode()
    else:
        test_plan_refinement()


if __name__ == "__main__":
    main()

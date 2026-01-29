#!/usr/bin/env python3
"""
Test script for Router Agent.

Tests document selection on sample questions.
"""
import argparse
from src.agents.router_agent import RouterAgent, RouterConfig


def test_router():
    """Test router agent with sample questions."""

    # Initialize router
    print("Initializing Router Agent...")
    router = RouterAgent(
        summaries_path="artifacts/document_summaries.json",
        config=RouterConfig(
            model="llama3.2",
            top_k_docs=3,
            temperature=0.0
        )
    )

    # Test questions
    test_cases = [
        {
            "id": "EF_1",
            "question": "What are the rules for an ineligible risk?",
            "expected_types": ["rules_manual", "checklist"],
            "top_k": 3
        },
        {
            "id": "EF_2",
            "question": "What is the premium for a Tier 1 building with Protection Class 5, Coverage A of $500,000 and 2% deductible?",
            "expected_types": ["rate_pages", "actuarial_exhibit"],
            "top_k": 3
        },
        {
            "id": "CUSTOM_1",
            "question": "What are the underwriting guidelines for coastal properties?",
            "expected_types": ["rules_manual", "checklist"],
            "top_k": 2
        },
        {
            "id": "CUSTOM_2",
            "question": "How are rate changes justified in the filing?",
            "expected_types": ["actuarial_memo", "actuarial_exhibit"],
            "top_k": 2
        }
    ]

    print(f"\n{'=' * 70}")
    print("ROUTER AGENT TEST SUITE")
    print(f"{'=' * 70}\n")

    results = []

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'─' * 70}")
        print(f"Test {i}/{len(test_cases)}: {test_case['id']}")
        print(f"{'─' * 70}")

        # Select documents
        selected = router.select_documents(
            question=test_case['question'],
            top_k=test_case['top_k'],
            verbose=True
        )

        # Check document types
        selected_types = []
        for doc in selected:
            summary = router.get_document_summary(doc)
            if summary:
                selected_types.append(summary.get('document_type', 'unknown'))

        # Evaluate
        expected = test_case['expected_types']
        type_match = any(t in selected_types for t in expected)

        result = {
            "test_id": test_case['id'],
            "question": test_case['question'],
            "selected_docs": selected,
            "selected_types": selected_types,
            "expected_types": expected,
            "type_match": type_match
        }
        results.append(result)

        # Print evaluation
        print(f"\nEvaluation:")
        print(f"  Expected types: {', '.join(expected)}")
        print(f"  Selected types: {', '.join(selected_types)}")
        print(f"  Match: {'✓ PASS' if type_match else '✗ FAIL'}")

    # Summary
    print(f"\n{'=' * 70}")
    print("TEST SUMMARY")
    print(f"{'=' * 70}")

    passed = sum(1 for r in results if r['type_match'])
    total = len(results)

    print(f"\nPassed: {passed}/{total} ({100 * passed / total:.0f}%)")

    print("\nDetailed Results:")
    for r in results:
        status = "✓" if r['type_match'] else "✗"
        print(f"  {status} {r['test_id']}: {r['question'][:60]}...")
        print(f"     Selected: {', '.join([d[:40] for d in r['selected_docs']])}")

    print()


def interactive_mode():
    """Interactive mode for testing router with custom questions."""

    print("Initializing Router Agent...")
    router = RouterAgent(
        summaries_path="artifacts/document_summaries.json",
        config=RouterConfig(model="llama3.2", top_k_docs=3)
    )

    print(f"\n{'=' * 70}")
    print("ROUTER AGENT - INTERACTIVE MODE")
    print(f"{'=' * 70}")
    print("Type your questions to see which documents are selected.")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            question = input("Question: ").strip()

            if question.lower() in ['quit', 'exit', 'q']:
                break

            if not question:
                continue

            # Route question
            selected = router.select_documents(
                question=question,
                top_k=3,
                verbose=True
            )

            print()  # Blank line before next question

        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}\n")


def main():
    parser = argparse.ArgumentParser(description="Test Router Agent")
    parser.add_argument(
        '--mode',
        choices=['test', 'interactive'],
        default='test',
        help='Test mode: run test suite or interactive'
    )

    args = parser.parse_args()

    if args.mode == 'test':
        test_router()
    else:
        interactive_mode()


if __name__ == "__main__":
    main()

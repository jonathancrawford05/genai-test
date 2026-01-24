#!/usr/bin/env python3
"""
Test script for Retriever Agent.

Tests full Router → Planner → Retriever pipeline.
"""
import argparse
from src.agents.router_agent import RouterAgent, RouterConfig
from src.agents.planner_agent import PlannerAgent, PlannerConfig
from src.agents.retriever_agent import RetrieverAgent, RetrieverConfig
from src.onnx_processor import ONNXProcessor


def test_full_pipeline():
    """Test complete Router → Planner → Retriever pipeline."""

    print("Initializing multi-agent pipeline...")
    print()

    # Initialize agents
    router = RouterAgent(
        summaries_path="artifacts/document_summaries.json",
        config=RouterConfig(model="llama3.2", top_k_docs=3)
    )

    planner = PlannerAgent(
        config=PlannerConfig(model="llama3.2", temperature=0.0)
    )

    # Initialize processor (reuse existing index)
    processor = ONNXProcessor(
        persist_directory="./chroma_db_onnx",
        collection_name="pdf_documents"
    )

    retriever = RetrieverAgent(
        processor=processor,
        config=RetrieverConfig(embedding_type="onnx", top_k_per_step=5)
    )

    # Test cases
    test_cases = [
        {
            "id": "EF_1",
            "question": "What are the rules for an ineligible risk?",
        },
        {
            "id": "EF_2",
            "question": "What is the premium for a Tier 1 building with Protection Class 5, Coverage A of $500,000 and 2% deductible?",
        }
    ]

    print(f"{'=' * 70}")
    print("FULL PIPELINE TEST SUITE")
    print(f"{'=' * 70}\n")

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'═' * 70}")
        print(f"Test {i}/{len(test_cases)}: {test_case['id']}")
        print(f"{'═' * 70}")

        question = test_case['question']
        print(f"Question: {question}\n")

        # Step 1: Router
        print(f"{'─' * 70}")
        print("[STEP 1] ROUTER - Document Selection")
        print(f"{'─' * 70}")

        selected_docs = router.select_documents(
            question=question,
            top_k=3,
            verbose=False
        )

        print(f"Selected {len(selected_docs)} documents:")
        for doc in selected_docs:
            doc_type = router.summaries.get(doc, {}).get('document_type', 'unknown')
            print(f"  - {doc} ({doc_type})")

        # Step 2: Planner
        print(f"\n{'─' * 70}")
        print("[STEP 2] PLANNER - Strategy Formulation")
        print(f"{'─' * 70}")

        plan = planner.create_plan(
            question=question,
            selected_documents=selected_docs,
            summaries=router.summaries,
            verbose=False
        )

        print(f"Strategy: {plan.strategy}")
        print(f"Steps: {len(plan.steps)}")
        print(f"Requires combination: {plan.requires_combination}")

        for step in plan.steps:
            print(f"\n  Step {step.step_number}: {step.description}")
            print(f"    Query: \"{step.query}\"")
            print(f"    Docs: {', '.join(step.target_documents)}")

        # Step 3: Retriever
        print(f"\n{'─' * 70}")
        print("[STEP 3] RETRIEVER - Plan Execution")
        print(f"{'─' * 70}")

        execution_result = retriever.execute_plan(plan, verbose=True)

        # Show retrieved information
        print(f"\n{'─' * 70}")
        print("RETRIEVED INFORMATION")
        print(f"{'─' * 70}")

        for step_result in execution_result.step_results:
            print(f"\nStep {step_result.step_number}: {step_result.num_chunks} chunks retrieved")
            for j, chunk in enumerate(step_result.chunks[:2], 1):  # Show top 2
                print(f"  [{j}] {chunk['source_file']} (page {chunk['page_number']})")
                print(f"      {chunk['text'][:150]}...")

        if execution_result.combined_context:
            print(f"\nCombined context: {len(execution_result.combined_context)} characters")

        # Get answer context
        context = retriever.get_answer_context(execution_result, max_chunks=3)

        print(f"\n{'─' * 70}")
        print("ANSWER CONTEXT (for LLM)")
        print(f"{'─' * 70}")
        print(f"Length: {len(context)} characters")
        print(f"\nPreview:")
        print(context[:500])
        print("...")

    print(f"\n{'═' * 70}")
    print("✓ Pipeline test complete")
    print(f"{'═' * 70}\n")


def interactive_mode():
    """Interactive mode for testing full pipeline."""

    print("Initializing multi-agent pipeline...")

    # Initialize agents
    router = RouterAgent(
        summaries_path="artifacts/document_summaries.json",
        config=RouterConfig(model="llama3.2", top_k_docs=3)
    )

    planner = PlannerAgent(
        config=PlannerConfig(model="llama3.2", temperature=0.0)
    )

    processor = ONNXProcessor(
        persist_directory="./chroma_db_onnx",
        collection_name="pdf_documents"
    )

    retriever = RetrieverAgent(
        processor=processor,
        config=RetrieverConfig(embedding_type="onnx", top_k_per_step=5)
    )

    print(f"\n{'=' * 70}")
    print("ROUTER → PLANNER → RETRIEVER INTERACTIVE MODE")
    print(f"{'=' * 70}")
    print("Type your questions to see the full pipeline in action.")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            question = input("Question: ").strip()

            if question.lower() in ['quit', 'exit', 'q']:
                break

            if not question:
                continue

            print(f"\n{'═' * 70}")

            # Route
            print("[ROUTER]")
            selected = router.select_documents(question=question, top_k=3, verbose=False)
            print(f"  Selected: {', '.join([d[:40] for d in selected])}")

            # Plan
            print("\n[PLANNER]")
            plan = planner.create_plan(
                question=question,
                selected_documents=selected,
                summaries=router.summaries,
                verbose=False
            )
            print(f"  Strategy: {plan.strategy}")
            print(f"  Steps: {len(plan.steps)}")
            for step in plan.steps:
                print(f"    {step.step_number}. {step.query}")

            # Retrieve
            print("\n[RETRIEVER]")
            result = retriever.execute_plan(plan, verbose=False)

            total_chunks = sum(r.num_chunks for r in result.step_results)
            print(f"  Retrieved: {total_chunks} chunks total")

            # Show context
            context = retriever.get_answer_context(result, max_chunks=3)
            print(f"\n  Context preview ({len(context)} chars):")
            print(f"  {context[:200]}...")

            print(f"{'═' * 70}\n")

        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}\n")


def test_retriever_only():
    """Test retriever with a pre-made plan."""

    print("Initializing Retriever...")

    processor = ONNXProcessor(
        persist_directory="./chroma_db_onnx",
        collection_name="pdf_documents"
    )

    retriever = RetrieverAgent(
        processor=processor,
        config=RetrieverConfig(embedding_type="onnx", top_k_per_step=5)
    )

    # Create a simple plan manually
    from src.agents.planner_agent import RetrievalPlan, RetrievalStep

    plan = RetrievalPlan(
        question="What are the rules for ineligible risks?",
        strategy="Direct lookup from rules manual",
        steps=[
            RetrievalStep(
                step_number=1,
                description="Search for ineligibility rules",
                target_documents=[
                    "(215066178-180449588)-CT MAPS Homeowner Rules Manual eff 08.18.25 v4.pdf"
                ],
                query="ineligible risk rules restrictions",
                expected_output="List of ineligibility criteria"
            )
        ],
        success_criteria="Find comprehensive list of ineligibility rules",
        requires_combination=False
    )

    print(f"\n{'=' * 70}")
    print("RETRIEVER-ONLY TEST")
    print(f"{'=' * 70}\n")

    print("Executing plan...")
    result = retriever.execute_plan(plan, verbose=True)

    print(f"\n{'─' * 70}")
    print("RESULTS")
    print(f"{'─' * 70}")

    for step_result in result.step_results:
        print(f"\nStep {step_result.step_number}: Retrieved {step_result.num_chunks} chunks")
        for i, chunk in enumerate(step_result.chunks[:3], 1):
            print(f"\n  Chunk {i}:")
            print(f"    Source: {chunk['source_file']}")
            print(f"    Page: {chunk['page_number']}")
            print(f"    Text: {chunk['text'][:200]}...")

    print()


def main():
    parser = argparse.ArgumentParser(description="Test Retriever Agent")
    parser.add_argument(
        '--mode',
        choices=['test', 'interactive', 'retriever-only'],
        default='test',
        help='Test mode'
    )

    args = parser.parse_args()

    if args.mode == 'test':
        test_full_pipeline()
    elif args.mode == 'interactive':
        interactive_mode()
    else:
        test_retriever_only()


if __name__ == "__main__":
    main()

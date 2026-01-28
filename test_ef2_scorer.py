#!/usr/bin/env python3
"""
Test the EF_2 component scorer with sample data.
"""

import json
from score_ef2_components import EF2Scorer, analyze_results
from pathlib import Path


def create_test_data():
    """Create sample test results."""

    # Perfect answer
    perfect_answer = """
    Based on Rule C-7 (page 23), since the property is more than 2,500 feet
    from the coastline in a Coastline Neighborhood, the mandatory hurricane
    deductible is 2%.

    From the CT Homeowners MAPS Rate Pages, Exhibit 1 (Page 4), the hurricane
    base rate is $293.

    Looking at Exhibit 6 (Page 71), for an HO3 policy with Coverage A of
    $750,000 and a 2% deductible, the factor is 2.061.

    Therefore, the hurricane premium is:
    $293 × 2.061 = $603.873, which rounds to $604
    """

    # Partial answer (missing factor and final calculation)
    partial_answer = """
    The property requires a 2% hurricane deductible according to Rule C-7
    since it is more than 2,500 feet from the coast.

    The hurricane base rate from Exhibit 1 is $293.

    The premium should be calculated using the appropriate factor.
    """

    # Poor answer (generic, no specifics)
    poor_answer = """
    The hurricane premium is calculated based on the policy's deductible
    requirements and base rates from the rating manual.
    """

    return [
        {
            "question_id": "EF_2",
            "question": "Calculate hurricane premium",
            "expected_output": "$604",
            "actual_output": perfect_answer,
            "variation_name": "chunk_2000_topk_5_window0_docs1_page",
            "config": {
                "variation_name": "chunk_2000_topk_5_window0_docs1_page",
                "router": {"top_k_docs": 1},
                "retriever": {
                    "top_k_per_step": 5,
                    "chunk_size": 2000,
                    "chunk_overlap": 0,
                    "expand_context": 0,
                    "chunking_strategy": "page",
                    "use_hybrid": True,
                    "hybrid_alpha": 0.3
                }
            },
            "router_selected_docs": [
                "(215066178-180449588)-CT MAPS Homeowner Rules Manual eff 08.18.25 v4.pdf",
                "(215004905-180407973)-CT Homeowners MAPS Rate Pages Eff 8.18.25 v3.pdf"
            ],
            "retrieval_steps": [
                {
                    "step_number": 1,
                    "chunks": [
                        {
                            "source_file": "(215066178-180449588)-CT MAPS Homeowner Rules Manual eff 08.18.25 v4.pdf",
                            "page_number": 23
                        },
                        {
                            "source_file": "(215004905-180407973)-CT Homeowners MAPS Rate Pages Eff 8.18.25 v3.pdf",
                            "page_number": 4
                        }
                    ]
                }
            ],
            "execution_time_seconds": 45.2,
            "total_chunks_retrieved": 5
        },
        {
            "question_id": "EF_2",
            "question": "Calculate hurricane premium",
            "expected_output": "$604",
            "actual_output": partial_answer,
            "variation_name": "chunk_1000_topk_3_window100_docs2_document",
            "config": {
                "variation_name": "chunk_1000_topk_3_window100_docs2_document",
                "router": {"top_k_docs": 2},
                "retriever": {
                    "top_k_per_step": 3,
                    "chunk_size": 1000,
                    "chunk_overlap": 100,
                    "expand_context": 100,
                    "chunking_strategy": "document",
                    "use_hybrid": True,
                    "hybrid_alpha": 0.5
                }
            },
            "router_selected_docs": [
                "(215066178-180449588)-CT MAPS Homeowner Rules Manual eff 08.18.25 v4.pdf",
                "(215004905-180407973)-CT Homeowners MAPS Rate Pages Eff 8.18.25 v3.pdf"
            ],
            "retrieval_steps": [
                {
                    "step_number": 1,
                    "chunks": [
                        {
                            "source_file": "(215066178-180449588)-CT MAPS Homeowner Rules Manual eff 08.18.25 v4.pdf",
                            "page_number": 23
                        }
                    ]
                }
            ],
            "execution_time_seconds": 38.5,
            "total_chunks_retrieved": 3
        },
        {
            "question_id": "EF_2",
            "question": "Calculate hurricane premium",
            "expected_output": "$604",
            "actual_output": poor_answer,
            "variation_name": "chunk_500_topk_10_window200_docs3_sliding_window",
            "config": {
                "variation_name": "chunk_500_topk_10_window200_docs3_sliding_window",
                "router": {"top_k_docs": 3},
                "retriever": {
                    "top_k_per_step": 10,
                    "chunk_size": 500,
                    "chunk_overlap": 200,
                    "expand_context": 200,
                    "chunking_strategy": "sliding_window",
                    "use_hybrid": False,
                    "hybrid_alpha": 0.0
                }
            },
            "router_selected_docs": [
                "(215066178-180449588)-CT MAPS Homeowner Rules Manual eff 08.18.25 v4.pdf",
                "(215004905-180407973)-CT Homeowners MAPS Rate Pages Eff 8.18.25 v3.pdf"
            ],
            "retrieval_steps": [
                {
                    "step_number": 1,
                    "chunks": [
                        {
                            "source_file": "(215066178-180449588)-CT MAPS Homeowner Rules Manual eff 08.18.25 v4.pdf",
                            "page_number": 10
                        }
                    ]
                }
            ],
            "execution_time_seconds": 52.1,
            "total_chunks_retrieved": 10
        }
    ]


def test_scorer():
    """Test the scorer with sample data."""
    print("="*80)
    print("TESTING EF_2 COMPONENT SCORER")
    print("="*80)

    # Create test data
    test_results = create_test_data()

    # Save to temporary file
    test_file = Path("test_results_ef2.json")
    with open(test_file, 'w') as f:
        json.dump(test_results, f, indent=2)

    print(f"\n✓ Created test file: {test_file}")
    print(f"  Test cases: {len(test_results)}")

    # Score individual results
    scorer = EF2Scorer()

    print("\n" + "-"*80)
    print("INDIVIDUAL RESULT SCORING")
    print("-"*80)

    for i, result in enumerate(test_results, 1):
        scores, reasoning = scorer.score_result(result)

        print(f"\nTest Case {i}: {result['variation_name']}")
        print(f"Total Score: {scores.total}/100")
        print(f"  - Document Retrieval: {scores.document_retrieval}/20")
        print(f"    {reasoning['document_retrieval']}")
        print(f"  - Deductible ID: {scores.deductible_identification}/20")
        print(f"    {reasoning['deductible_identification']}")
        print(f"  - Base Rate ID: {scores.base_rate_identification}/20")
        print(f"    {reasoning['base_rate_identification']}")
        print(f"  - Factor ID: {scores.factor_identification}/20")
        print(f"    {reasoning['factor_identification']}")
        print(f"  - Calculation: {scores.calculation}/20")
        print(f"    {reasoning['calculation']}")

    # Test full analysis
    print("\n" + "-"*80)
    print("FULL ANALYSIS TEST")
    print("-"*80)

    df = analyze_results(str(test_file), output_csv="test_ef2_scores.csv")

    print(f"\n✓ Analyzed {len(df)} results")
    print(f"✓ Saved to test_ef2_scores.csv")

    print("\nResults Summary:")
    print(df[['variation_name', 'chunk_size', 'top_k', 'total_score']].to_string(index=False))

    # Cleanup
    test_file.unlink()
    print(f"\n✓ Cleaned up test file")

    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)
    print("\nExpected results:")
    print("  Test Case 1 (perfect answer): ~100/100")
    print("  Test Case 2 (partial answer): ~55-65/100")
    print("  Test Case 3 (poor answer): ~20-30/100")
    print("\nIf scores match expectations, the scorer is working correctly!")


if __name__ == "__main__":
    test_scorer()

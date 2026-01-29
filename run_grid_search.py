"""
Run comprehensive grid search with enhanced observability.

Uses your 270-variation grid search with detailed JSON output.
"""
from experiment_runner_enhanced import EnhancedExperimentRunner, VariationConfig
from src.agents import OrchestratorConfig, RouterConfig, PlannerConfig, RetrieverConfig


class GridSearchRunner(EnhancedExperimentRunner):
    """Your 270-variation grid search with hybrid search enabled."""

    def create_variations(self):
        """
        Create 270 variations testing:
        - chunk_size: 400, 800, 1200, 1600, 2000 (5 values)
        - top_k_per_step: 1, 3, 5 (3 values)
        - chunking_strategy: page, document (2 values)
        - top_k_docs: 1, 3, 5 (3 values)
        - expand_context: 0, 1, 2 (3 values)

        Total: 5 * 3 * 2 * 3 * 3 = 270 variations

        All variations use hybrid search (confirmed best performer: 47% vs ~10%).
        """
        variations = []

        for i in range(5):
            chunk_size = (i + 1) * 400

            for top_k_per_step in range(1, 6, 2):  # 1, 3, 5
                for typ in ["page", "document"]:
                    for docs in range(1, 6, 2):  # 1, 3, 5
                        for wdw in [0, 1, 2]:
                            variations.append(
                                VariationConfig(
                                    name=f"chunk_{chunk_size}_topk_{top_k_per_step}_window{wdw}_docs{docs}_{typ}",
                                    description=(
                                        f"chunk_size={chunk_size}, "
                                        f"top_k_per_step={top_k_per_step}, "
                                        f"top_k_docs={docs}, "
                                        f"chunking_strategy={typ}, "
                                        f"expand_context={wdw}"
                                    ),
                                    orchestrator_config=OrchestratorConfig(
                                        model="llama3.2",
                                        temperature=0.0,
                                        max_answer_tokens=4096,
                                        router_config=RouterConfig(
                                            model="llama3.2",
                                            top_k_docs=docs,
                                            temperature=0.0,
                                        ),
                                        planner_config=PlannerConfig(
                                            model="llama3.2",
                                            temperature=0.0,
                                        ),
                                        retriever_config=RetrieverConfig(
                                            top_k_per_step=top_k_per_step,
                                            chunk_size=chunk_size,
                                            chunk_overlap=200,
                                            expand_context=wdw,
                                            chunking_strategy=typ,
                                            use_hybrid=True,  # ENABLED - confirmed best
                                            hybrid_alpha=0.5  # Adaptive at runtime
                                        ),
                                    ),
                                )
                            )

        return variations


if __name__ == "__main__":
    print("="*70)
    print("GRID SEARCH: 270 Variations")
    print("="*70)
    print("\nParameters tested:")
    print("  - chunk_size: 400, 800, 1200, 1600, 2000")
    print("  - top_k_per_step: 1, 3, 5")
    print("  - chunking_strategy: page, document")
    print("  - top_k_docs: 1, 3, 5")
    print("  - expand_context: 0, 1, 2")
    print("  - use_hybrid: True (all variations)")
    print("\nTotal combinations: 270")
    print("Questions: 2 (EF_1, EF_2)")
    print("Total runs: 540")
    print("\n⏱️  Estimated time: 10-20 hours")
    print("\nOutput:")
    print("  - results/experiment_results_<timestamp>.csv")
    print("  - results/detailed_results_<timestamp>.json (with chunk indices)")
    print("  - results/summary_report_<timestamp>.md")
    print("="*70)

    confirm = input("\nProceed with full grid search? (yes/no): ")

    if confirm.lower() == 'yes':
        runner = GridSearchRunner()
        results_df = runner.run_all_experiments(
            verbose=False,  # Set True to see all agent decisions
            clean_collections=True,
            save_detailed_json=True
        )
        runner.generate_report(results_df)

        print("\n" + "="*70)
        print("✓ GRID SEARCH COMPLETE")
        print("="*70)
        print("\nNext steps:")
        print("  1. Open experiments.ipynb to analyze results")
        print("  2. Load results/detailed_results_<timestamp>.json")
        print("  3. Compare chunk indices for best/worst performers")
        print("  4. Identify optimal parameter ranges")
    else:
        print("\nCancelled. Use experiments.ipynb for subset testing first.")

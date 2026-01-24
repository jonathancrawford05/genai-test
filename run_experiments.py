"""
Main script to run Part 2 experiments.

Usage:
    python run_experiments.py
"""
from experiment_runner import ExperimentRunner


def main():
    print("\n" + "="*70)
    print("GenAI Test - Part 2: Experimentation Harness")
    print("="*70)
    print("\nThis will run 4 variations on 2 test questions (8 total runs)")
    print("\nVariations:")
    print("  1. baseline - ONNX embeddings, top_k_docs=3, top_k_per_step=5")
    print("  2. ollama_embeddings - Higher quality embeddings")
    print("  3. high_depth - More documents and chunks (5 docs, 10 chunks)")
    print("  4. conservative - Fewer documents and chunks (2 docs, 3 chunks)")
    print("\nStarting experiments...")
    print("="*70 + "\n")

    # Run experiments
    runner = ExperimentRunner(questions_csv="artifacts/questions.csv")
    results_df = runner.run_all_experiments(verbose=True)

    # Generate report
    runner.generate_report(results_df, output_dir="results")

    print("\n" + "="*70)
    print("✓ Experiments complete!")
    print("="*70)
    print("\nResults saved to:")
    print("  - results/experiment_results_*.csv (detailed)")
    print("  - results/summary_report_*.md (summary)")
    print()


if __name__ == "__main__":
    main()

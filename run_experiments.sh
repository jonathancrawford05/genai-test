#!/bin/bash
# Wrapper script to run experiments with all embedding types

echo "============================================================"
echo "RAG EXPERIMENTATION SUITE"
echo "============================================================"
echo ""
echo "This will run experiments with:"
echo "  - Embeddings: ONNX (79MB, 384 dims), Ollama (274MB, 768 dims, RAG-optimized)"
echo "  - LLMs: phi3, llama3.2, llama3.1"
echo "  - Top-K: 3, 5, 10"
echo ""
echo "Note: This may take 30-60 minutes depending on your system."
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    exit 1
fi

# Run enhanced experiments
python enhanced_experiment.py \
  --embeddings onnx nomic-embed-text \
  --llms phi3 llama3.2 llama3.1 \
  --top-k 3 5 10 \
  --output-json experiment_results.json \
  --output-csv experiment_results_detailed.csv

echo ""
echo "============================================================"
echo "✓ Experiments complete!"
echo "============================================================"
echo ""
echo "Results saved to:"
echo "  - experiment_results.json"
echo "  - experiment_results_detailed.csv"
echo ""
echo "Open experiment_notebook.ipynb in Jupyter to analyze results."
echo ""

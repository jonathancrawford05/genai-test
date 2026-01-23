#!/bin/bash
# Clear all ChromaDB databases to start fresh

echo "============================================================"
echo "CLEAR ALL CHROMADB DATABASES"
echo "============================================================"
echo ""
echo "This will delete:"
echo "  - chroma_db_onnx/"
echo "  - chroma_db_ollama/"
echo ""
echo "You will need to re-index your PDFs after this."
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 1
fi

# Remove databases
if [ -d "chroma_db_onnx" ]; then
    echo "Removing chroma_db_onnx..."
    rm -rf chroma_db_onnx
fi

if [ -d "chroma_db_ollama" ]; then
    echo "Removing chroma_db_ollama..."
    rm -rf chroma_db_ollama
fi

# Remove any legacy databases
if [ -d "chroma_db_light" ]; then
    echo "Removing chroma_db_light (legacy)..."
    rm -rf chroma_db_light
fi

if [ -d "chroma_db_sentence_transformers" ]; then
    echo "Removing chroma_db_sentence_transformers (legacy)..."
    rm -rf chroma_db_sentence_transformers
fi

echo ""
echo "✓ All databases cleared!"
echo ""
echo "Next steps:"
echo "  1. Run experiments: python fixed_experiment.py"
echo "  2. Or index manually: python index_light.py artifacts/1"
echo ""

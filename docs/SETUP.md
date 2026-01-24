# Setup Instructions

## Prerequisites

- **Python**: 3.11+
- **Poetry**: For dependency management
- **Ollama**: For LLM inference and embeddings

## Installation

### 1. Install Poetry

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

### 2. Install Dependencies

```bash
cd genai-test
poetry install
```

This installs all required packages:
- `ollama` - LLM inference
- `chromadb` - Vector database
- `pypdf` - PDF parsing
- `onnxruntime` - ONNX embeddings
- `pandas` - Data analysis

### 3. Install and Start Ollama

**macOS:**
```bash
brew install ollama
ollama serve
```

**Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve
```

**Windows:**
Download from https://ollama.com/download

### 4. Pull Required Models

In a separate terminal:

```bash
# Pull LLM model
ollama pull llama3.2

# Pull embedding model (for Ollama embeddings variation)
ollama pull nomic-embed-text
```

## Generating Document Summaries

The system requires document summaries to be generated first.

```bash
# Generate summaries for all PDFs in artifacts/1
poetry run python -m src.agents.document_summarizer \
    --pdf-folder artifacts/1 \
    --output artifacts/document_summaries.json
```

This takes ~10-15 minutes for 22 documents.

**Note:** `artifacts/document_summaries.json` is already included in the repository.

## Running Part 1: PDF Question Answering

### Single Question

```python
from answer_pdf_question import answer_pdf_question

answer = answer_pdf_question(
    question="List all rating plan rules",
    pdfs_folder="artifacts/1",
    verbose=True  # Show intermediate steps
)

print(answer)
```

### Test Script

```bash
poetry run python test_orchestrator.py
```

### Example Script

```bash
poetry run python answer_pdf_question.py
```

## Running Part 2: Experimentation Harness

```bash
poetry run python run_experiments.py
```

This will:
1. Run 4 variations on 2 test questions (8 total runs)
2. Calculate accuracy and performance metrics
3. Generate comparison reports in `results/` folder

**Expected runtime:** 10-20 minutes total

## Project Structure

```
genai-test/
├── answer_pdf_question.py     # Part 1: Main interface
├── experiment_runner.py        # Part 2: Experiment framework
├── run_experiments.py          # Part 2: Main execution
├── test_orchestrator.py        # Quick test script
├── src/
│   ├── agents/                 # Multi-agent system
│   │   ├── base_agent.py
│   │   ├── router_agent.py     # Document selection
│   │   ├── planner_agent.py    # Strategy formulation
│   │   ├── retriever_agent.py  # Chunk retrieval
│   │   └── orchestrator_agent.py  # Pipeline coordination
│   ├── onnx_processor.py       # ONNX embeddings
│   └── ollama_processor.py     # Ollama embeddings
├── artifacts/
│   ├── questions.csv           # Test questions
│   ├── document_summaries.json # Pre-generated summaries
│   └── 1/                      # PDF documents
├── results/                    # Experiment outputs
└── docs/
    ├── architecture.md         # System architecture
    ├── SETUP.md               # This file
    ├── DECISIONS.md           # Design decisions
    └── future_refinements.md  # Improvement ideas
```

## Troubleshooting

### "Failed to connect to Ollama"

Ensure Ollama is running:
```bash
ollama serve
```

### "Collection has wrong dimensions"

Clear ChromaDB collections:
```bash
rm -rf chroma_db_onnx chroma_db_ollama
```

Re-run to rebuild indexes.

### "Document summaries not found"

Generate summaries:
```bash
poetry run python -m src.agents.document_summarizer \
    --pdf-folder artifacts/1 \
    --output artifacts/document_summaries.json
```

### Slow performance

- **Use ONNX embeddings** (default) - 2x faster than Ollama
- **Reduce top_k** - Fewer documents/chunks = faster
- **Use SSD** - ChromaDB benefits from fast disk I/O

## Verification

Test that everything is working:

```bash
# 1. Check Ollama
ollama list
# Should show: llama3.2, nomic-embed-text

# 2. Run quick test
poetry run python test_orchestrator.py
# Should complete without errors in ~30-60 seconds

# 3. Run full experiments
poetry run python run_experiments.py
# Should complete in 10-20 minutes
```

## Cost Estimation

**Using Ollama (local inference):**
- Cost: $0 (runs on your machine)
- Requirements: 8GB+ RAM, 10GB disk space

**Alternative: Using OpenAI API:**
- Not currently implemented
- Would cost ~$0.50-$2 per experiment run
- Would require API key and code modifications

## Next Steps

1. ✅ Run Part 1 to verify system works
2. ✅ Run Part 2 experiments
3. ✅ Review results in `results/` folder
4. 📊 Analyze which variation performs best
5. 🔧 Iterate based on findings

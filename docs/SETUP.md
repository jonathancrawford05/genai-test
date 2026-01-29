# Setup Instructions

## Prerequisites

- **Python**: 3.11+
- **Poetry**: For dependency management
- **Ollama**: For LLM inference (llama3.2)

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
- `ollama` - LLM inference (llama3.2)
- `chromadb` - Vector database with ONNX embeddings
- `pypdf` - PDF text extraction
- `rank-bm25` - Keyword search for hybrid retrieval
- `pandas` - Data analysis
- `langchain-*` - LLM orchestration

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
# Pull LLM model for inference (used by all agents)
ollama pull llama3.2

# Pull LLM model for document summarization (larger model for better summaries)
ollama pull gpt-oss:20b
```

**Note:** `gpt-oss:20b` is only needed if you want to regenerate document summaries. The repository already includes pre-generated summaries in `artifacts/document_summaries.json`.

## Generating Document Summaries (Optional)

The system requires document summaries for the Router agent. Pre-generated summaries are already included in `artifacts/document_summaries.json`.

If you need to regenerate summaries:

```bash
# Uses gpt-oss:20b model for high-quality summaries
poetry run python -m src.agents.document_summarizer \
    --pdf-folder artifacts/1 \
    --output artifacts/document_summaries.json
```

**Requirements:**
- Requires `gpt-oss:20b` model (see step 4 above)
- Takes ~10-15 minutes for 22 documents
- Uses larger model (20B parameters) for better summary quality

**Note:** This step is optional - the repository already includes pre-generated summaries.

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
1. Run 6 variations on 2 test questions (12 total runs)
2. Test different configurations:
   - **Baseline**: Standard document-level chunking
   - **High Depth**: Increased retrieval (top_k=10)
   - **Conservative**: Reduced retrieval (top_k=3)
   - **Sliding Window**: Document-level + ±2 chunk expansion
   - **Page Window**: Page-level chunking + expansion
   - **Hybrid Search**: BM25 + semantic with expansion
3. Calculate accuracy and performance metrics
4. Generate comparison reports in `results/` folder

**Expected runtime:** 15-30 minutes total

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
│   │   ├── planner_agent.py    # Strategy formulation (enumeration-aware)
│   │   ├── retriever_agent.py  # Hybrid retrieval + sliding window
│   │   ├── orchestrator_agent.py  # Pipeline coordination
│   │   └── document_summarizer.py # PDF summarization
│   ├── base_processor.py       # Base processor (configurable chunking)
│   ├── onnx_processor.py       # ONNX embeddings
│   └── hybrid_retriever.py     # BM25 + semantic hybrid search
├── artifacts/
│   ├── questions.csv           # Test questions
│   ├── document_summaries.json # Pre-generated summaries
│   └── 1/                      # PDF documents
├── chroma_db/                  # ChromaDB persistent storage
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
rm -rf chroma_db
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

- **Reduce top_k** - Fewer documents/chunks = faster retrieval
- **Use SSD** - ChromaDB benefits from fast disk I/O
- **Reduce chunk_size** - Smaller chunks = faster processing

## Verification

Test that everything is working:

```bash
# 1. Check Ollama
ollama list
# Should show: llama3.2

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

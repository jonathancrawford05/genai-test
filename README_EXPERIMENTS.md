# Experimentation Harness Guide

Complete guide for running experiments with your RAG system.

## Quick Start

**Run a quick test** (phi3 + llama3.2, ONNX embeddings):
```bash
python experiment.py --quick
```

**Expected time**: 2-5 minutes for 2 questions × 2 models = 4 experiments

## Full Experiments

### Test All Your Installed Models

```bash
python experiment.py \
  --embeddings onnx nomic-embed-text \
  --llms phi3 llama3.2 llama3 llama3.1 qwen3:14b \
  --top-k 3 5 10
```

**This runs**: 2 embeddings × 5 LLMs × 3 top_k values = **30 experiments**

### Test Specific Combinations

**Compare small vs large models:**
```bash
python experiment.py --llms phi3 qwen3:14b
```

**Test retrieval parameters:**
```bash
python experiment.py --llms llama3.2 --top-k 3 5 7 10
```

**Test embedding impact:**
```bash
python experiment.py --embeddings onnx nomic-embed-text --llms llama3.2
```

## What Gets Tested

### Embedding Models

1. **onnx** (ChromaDB default)
   - Size: ~79MB
   - Speed: Fast
   - Quality: Good for keyword matching
   - Index: ./chroma_db_onnx/

2. **nomic-embed-text** (Ollama)
   - Size: 274MB
   - Speed: Medium
   - Quality: Excellent for semantic search
   - Index: ./chroma_db_nomic-embed-text/

### LLM Models (Your Installed Models)

| Model | Size | Speed | Use Case |
|-------|------|-------|----------|
| **phi3** | 2.2GB | Fastest | Simple questions |
| **llama3.2** | 2.0GB | Fast | General purpose |
| **llama3** | 4.7GB | Medium | Better reasoning |
| **llama3.1** | 4.9GB | Medium | Latest improvements |
| **qwen3:14b** | 9.3GB | Slower | Best quality |
| **gpt-oss:20b** | 13GB | Slowest | Highest quality |

### Metrics Tracked

1. **Exact Match**: Does answer exactly match expected?
2. **Token F1**: Overlap between answer and expected (0-1)
3. **Latency**: Response time in seconds
4. **Model Size**: GB (proxy for compute requirements)

## Output

### During Execution

```
============================================================
EXPERIMENT: onnx_phi3_k5
============================================================

Setting up embedding: onnx
✓ Using existing index: 1523 chunks

Testing 2 questions with phi3...

[1/2] EF_1: ✓ (2.34s)
[2/2] EF_2: F1=0.95 (1.89s)
```

### Results Summary

```
RESULTS SUMMARY
================================================================================
Config                                   Exact    Avg F1   Avg Lat    Size(GB)
--------------------------------------------------------------------------------
onnx_phi3_k5                             1/2      0.975    2.12s      2.2
onnx_llama3.2_k5                         2/2      1.000    2.87s      2.0
onnx_qwen3:14b_k5                        2/2      1.000    5.43s      9.3

BEST PERFORMERS
================================================================================

Best Accuracy: onnx_llama3.2_k5
  Avg F1: 1.000

Fastest: onnx_phi3_k5
  Avg latency: 2.12s

Best Efficiency (F1/latency): onnx_llama3.2_k5

PER-QUESTION OPTIMAL MODEL
================================================================================

EF_1:
  Best: qwen3:14b (F1=1.000, 5.87s, 9.3GB)
  Optimal: llama3.2 (F1=1.000, 2.45s, 2.0GB) ← Use this!

EF_2:
  Best: llama3.2 (F1=1.000, 1.98s, 2.0GB) ← Already optimal!
```

### JSON Output

Results saved to `experiment_results.json`:

```json
[
  {
    "question_id": "EF_1",
    "question": "List all rating plan rules",
    "expected": "* Limits of Liability...",
    "answer": "* Limits of Liability...",
    "exact_match": true,
    "token_f1": 1.0,
    "latency_sec": 2.45,
    "config": "onnx_llama3.2_k5",
    "embedding_model": "onnx",
    "llm_model": "llama3.2",
    "llm_size_gb": 2.0,
    "top_k": 5
  },
  ...
]
```

## Analysis Features

### 1. Overall Best Config
Shows which combination performs best overall.

### 2. Speed vs Quality Tradeoffs
Identifies fastest model and most accurate model.

### 3. Per-Question Optimization
**Key feature**: Shows when smaller models are sufficient!

Example:
- Question EF_1: qwen3:14b gets F1=1.0 but so does llama3.2
- **Recommendation**: Use llama3.2 (2GB, 2.5s) instead of qwen3:14b (9.3GB, 5.9s)
- **Savings**: 4.3x smaller, 2.4x faster, same quality!

### 4. Efficiency Metric
F1 / latency ratio - best quality per second of wait time.

## Adding More Questions

**Simply add to `artifacts/questions.csv`:**

```csv
id,question,expected_output,PDF Folder
EF_1,List all rating plan rules,"* Rule 1...",1
EF_2,Calculate premium,$604,1
NEW_3,What is the coverage limit?,$750000,1
NEW_4,List all discounts,"* Protective Device...",1
```

The harness automatically picks up all questions!

## Tips for Experiments

### Start Small
```bash
# Test 2 models first
python experiment.py --quick
```

### Then Scale Up
```bash
# Add more models
python experiment.py --llms phi3 llama3.2 llama3 llama3.1

# Add embedding comparison
python experiment.py --embeddings onnx nomic-embed-text --llms llama3.2

# Full grid search (takes longer!)
python experiment.py --llms phi3 llama3.2 llama3.1 qwen3:14b --top-k 3 5 10
```

### Focus on Your Use Case

**For simple questions (like calculations):**
```bash
python experiment.py --llms phi3 llama3.2
```
Hypothesis: Small models should be sufficient.

**For complex questions (like lists):**
```bash
python experiment.py --llms llama3.1 qwen3:14b
```
Hypothesis: Larger models may perform better.

## Expected Insights

Based on your models, you might find:

1. **phi3** (2.2GB):
   - ✅ Fastest
   - ✅ Good for simple factual questions
   - ⚠️ May struggle with complex reasoning

2. **llama3.2** (2.0GB):
   - ✅ Best balance of speed/quality
   - ✅ Likely optimal for most questions
   - Sweet spot for production

3. **llama3.1** (4.9GB):
   - ✅ Better reasoning than llama3.2
   - ⚠️ Slower, may not justify cost for simple questions

4. **qwen3:14b** (9.3GB):
   - ✅ Highest quality
   - ⚠️ 2-3x slower than smaller models
   - ⚠️ Overkill for simple questions

5. **nomic-embed-text** vs **onnx**:
   - Semantic search vs keyword search
   - Worth testing if keyword matching fails

## Troubleshooting

### "Model not found"
```bash
# Pull missing models
ollama pull phi3
ollama pull llama3.2
ollama pull nomic-embed-text
```

### "Ollama connection refused"
```bash
# Start Ollama server
ollama serve
```

### Experiments taking too long
```bash
# Use --quick mode
python experiment.py --quick

# Or test fewer models
python experiment.py --llms phi3 llama3.2
```

### Out of disk space
```bash
# Clean up old experiment databases
rm -rf chroma_db_*

# They'll be rebuilt as needed
```

## Next Steps

1. **Run quick test** to verify everything works
2. **Add more questions** to questions.csv
3. **Run full experiments** with all models
4. **Analyze results** to find optimal configs
5. **Use insights** to configure production system

Example production config based on results:
```python
# If llama3.2 performs well on most questions:
PRODUCTION_CONFIG = {
    "embedding": "onnx",  # Fast enough
    "llm": "llama3.2",    # Best balance
    "top_k": 5,           # Sufficient context
}

# For critical/complex questions, upgrade to:
COMPLEX_CONFIG = {
    "embedding": "nomic-embed-text",  # Better semantic search
    "llm": "qwen3:14b",               # Best quality
    "top_k": 7,                        # More context
}
```

Happy experimenting! 🚀

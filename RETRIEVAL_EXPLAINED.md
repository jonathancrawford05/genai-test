# Retrieval Mechanisms Explained

Clear explanation of how document retrieval works in each configuration.

## Quick Summary

| Component | Method | Description |
|-----------|--------|-------------|
| **ONNX (UltraLightProcessor)** | Semantic similarity | ONNX-optimized all-MiniLM-L6-v2, 79MB, cosine distance |
| **nomic-embed-text (OllamaEmbeddingProcessor)** | Semantic similarity | Ollama embeddings, 274MB, RAG-optimized, cosine distance |

**None of these use BM25 or keyword matching!**

**Note:** Sentence-Transformers was removed from experiments as it's redundant with ONNX (same underlying model, just PyTorch vs ONNX runtime).

## Detailed Explanation

### 1. ONNX Embeddings (UltraLightProcessor)

**What it is:**
- ChromaDB's default embedding function
- Uses `all-MiniLM-L6-v2` model compiled to ONNX format
- Downloads 79MB model on first use

**How it works:**
```python
processor = UltraLightProcessor(...)
results = processor.query("What are the rules?", top_k=5)
```

1. **Text → Vector**: Query text converted to 384-dimensional vector
2. **Compare**: Cosine distance calculated between query vector and all document vectors
3. **Rank**: Documents sorted by similarity (lowest distance = most similar)
4. **Return**: Top K most similar chunks

**Retrieval method:** **Semantic similarity** (NOT keyword matching)
- "What are the rules?" matches "rating plan rules" even without exact words
- Uses meaning, not just keywords

**Performance:**
- Fast (ONNX optimized for CPU)
- Low memory (~200MB)
- Good semantic understanding

### 2. Sentence-Transformers (SentenceTransformerProcessor)

**What it is:**
- PyTorch-based `all-MiniLM-L6-v2` model
- Same model as ONNX, but PyTorch implementation
- More memory-intensive

**How it works:**
```python
processor = SentenceTransformerProcessor(model_name="all-MiniLM-L6-v2", ...)
results = processor.query("What are the rules?", top_k=5)
```

1. **Text → Vector**: Same 384-dimensional embeddings
2. **Compare**: Cosine distance (same math as ONNX)
3. **Rank**: By similarity
4. **Return**: Top K chunks

**Retrieval method:** **Semantic similarity** (identical to ONNX conceptually)

**Key difference from ONNX:**
- Same model architecture
- Different runtime (PyTorch vs ONNX)
- Should produce **very similar** (possibly identical) results
- Slower indexing (no ONNX optimization)
- Higher memory usage

**Performance:**
- Slower indexing (2-3x vs ONNX)
- Higher memory (1-2GB during indexing)
- Virtually identical retrieval quality

### 3. nomic-embed-text (OllamaEmbeddingProcessor) ✅

**What it is:**
- Ollama's embedding model (274MB)
- Optimized specifically for RAG tasks
- Different architecture from MiniLM
- Now integrated in `fixed_experiment.py`

**How it works:**
```python
processor = OllamaEmbeddingProcessor(model_name="nomic-embed-text", ...)
results = processor.query("What are the rules?", top_k=5)
```

1. **Text → Vector**: Query text converted via Ollama API
2. **Compare**: Cosine distance in embedding space
3. **Rank**: By similarity
4. **Return**: Top K chunks

**Retrieval method:** **Semantic similarity**
- Uses different embedding model than ONNX
- May produce different rankings than MiniLM
- Still cosine distance in vector space

**Why use this over ONNX:**
- Different model architecture (not just runtime difference)
- Specifically optimized for RAG retrieval tasks
- May retrieve more relevant chunks for question-answering

## NOT Used: BM25

**What is BM25?**
- Keyword-based ranking algorithm
- Used in the original `pdf_qa_prototype.py`
- Ranks based on term frequency and document frequency

**Why not BM25 in current system?**
- ChromaDB uses semantic embeddings by default
- Semantic search generally better for complex queries
- BM25 available but not implemented

**If you wanted BM25:**
```python
# Would need custom implementation or library like rank_bm25
from rank_bm25 import BM25Okapi
```

## Why Your Observations Make Sense

### Observation 1: Identical F1 Scores

**ONNX vs nomic-embed-text** showing identical scores is explained by:

**BUG**: The experiment script wasn't actually using different embeddings!
- Both used ONNX (just different database directories)
- Same embeddings → same retrieval → same answers → same F1

### Observation 2: Similar Indexing Times

**Also explained by the bug:**
- Both were using ONNX (fast)
- Not actually loading sentence-transformers (slow)

### What SHOULD Happen (Fixed)

With `fixed_experiment.py`:

**ONNX indexing:**
```
Using: ChromaDB ONNX embeddings (all-MiniLM-L6-v2, 79MB)
Indexing PDFs...
✓ Indexed 1523 chunks in 62.3s
```

**nomic-embed-text indexing:**
```
Using: Ollama nomic-embed-text embeddings (RAG-optimized, 274MB)
Testing Ollama connection with nomic-embed-text...
✓ Ollama connected (dim=768)
Indexing PDFs...
✓ Indexed 1523 chunks in XX.Xs  ← May differ from ONNX
```

**F1 Scores:**
- SHOULD differ (different model architectures!)
- ONNX uses all-MiniLM-L6-v2 (384 dims)
- nomic-embed-text uses different architecture (768 dims, RAG-optimized)
- Check `chunk_X_source` columns to see if retrieval differs

## Testing the Fix

### Run Fixed Experiments

```bash
python fixed_experiment.py \
  --embeddings onnx nomic-embed-text \
  --llms phi3 llama3.2 \
  --top-k 5
```

### What to Look For

**1. Indexing Time**
```
Setting up embedding: onnx
Using: ChromaDB ONNX embeddings (all-MiniLM-L6-v2, 79MB)
✓ Indexed 1523 chunks in 62.3s

Setting up embedding: nomic-embed-text
Using: Ollama nomic-embed-text embeddings (RAG-optimized, 274MB)
Testing Ollama connection with nomic-embed-text...
✓ Ollama connected (dim=768)
✓ Indexed 1523 chunks in XX.Xs  ← Timing may differ
```

**2. Memory Usage**
- ONNX: ~500MB peak
- nomic-embed-text: ~800MB-1GB peak (Ollama + embeddings)

**3. CSV Columns**
Open `experiment_fixed_detailed.csv`:
```
embedding_model,chunk_1_source,chunk_2_source,...
onnx,file1.pdf (p12),file2.pdf (p5),...
nomic-embed-text,file3.pdf (p8),file1.pdf (p15),...
                 ↑ These SHOULD differ!
```

**If chunks differ (expected):**
- Different model architectures (384 vs 768 dims)
- Different training objectives (general vs RAG-optimized)
- This means nomic-embed-text may retrieve different (hopefully better) chunks!

**If chunks are identical (unexpected):**
- Would indicate both models rank chunks identically
- Unlikely given different architectures
- Worth investigating if this happens

## Expected Results

### ONNX vs nomic-embed-text Comparison

These should produce **different** results because:

**ONNX (all-MiniLM-L6-v2):**
1. General-purpose sentence embedding model
2. 384-dimensional embeddings
3. Trained on diverse text similarity tasks
4. Fast, lightweight (79MB)

**nomic-embed-text:**
1. **Different architecture** (not just runtime)
2. **768-dimensional embeddings** (2x larger)
3. **RAG-optimized** training objective
4. Designed specifically for retrieval tasks
5. Slightly heavier (274MB)

**This means:** Results SHOULD differ!
- Different chunks may be retrieved
- F1 scores may differ
- One may outperform the other for RAG tasks

### When to Use Each

**Use ONNX if:**
- You need maximum speed and minimum memory
- General semantic similarity is sufficient
- 384 dimensions provide enough granularity

**Use nomic-embed-text if:**
- You want embeddings optimized for retrieval
- Higher dimensional space may help
- You're okay with slightly higher memory usage
- You want to test if RAG-specific training helps

**Test both!** That's what the experimentation harness is for.

## Summary

**Current implementations:**
1. ✅ **ONNX (UltraLightProcessor)**: Semantic similarity, all-MiniLM-L6-v2, 79MB, 384 dims
2. ✅ **nomic-embed-text (OllamaEmbeddingProcessor)**: Semantic similarity, RAG-optimized, 274MB, 768 dims
3. ❌ **Sentence-Transformers**: Removed (redundant with ONNX - same model, different runtime)

**Retrieval method (all):** **Semantic similarity** via embeddings + cosine distance

**NOT BM25**, NOT keyword matching (unless you specifically implement it).

**Bug identified:** ✅ Fixed in `fixed_experiment.py`

**What changed:**
- Script now properly swaps between embedding types
- Removed redundant sentence-transformers
- Added Ollama nomic-embed-text integration

**How to verify:**
- Check indexing times (may differ between ONNX and Ollama)
- Compare chunk sources in CSV (should differ - different models!)
- Compare F1 scores (may differ - RAG-optimized vs general-purpose)

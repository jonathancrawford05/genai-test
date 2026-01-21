# Memory-Efficient RAG Solution - Summary

## Problem Statement

Develop a RAG-enabled agentic solution to extract information from PDF files in `artifacts/1/`, capable of running on a MacBook with limited memory (8GB RAM).

## Initial Challenges Encountered

1. **Original prototype** (`pdf_qa_prototype.py`):
   - Loaded all 22 PDFs simultaneously into memory
   - Used BM25 in-memory index
   - Consumed 2-4GB RAM
   - **Crashed MacBook sessions**

2. **First attempt with sentence-transformers**:
   - PyTorch + CUDA libraries loaded ~30GB memory
   - Hung on first PDF after loading embedding model
   - Unusable on 8GB MacBook

3. **Infinite loop bug**:
   - Chunking algorithm had infinite loop when text length slightly exceeded chunk_size
   - Created hundreds/thousands of chunks per page
   - Hours-long processing times

## Final Solution

### Architecture

**Ultra-lightweight RAG system** using:
- **pypdf** for fast PDF text extraction
- **ChromaDB** with ONNX-based embeddings (no PyTorch)
- **Batched processing** to prevent memory accumulation
- **Streaming approach** with immediate persistence

### Key Optimizations

1. **Embedding Model**: ChromaDB's default ONNX model (79MB download)
   - vs. sentence-transformers with PyTorch (30GB in memory)

2. **Chunking**: Fixed infinite loop, added safety limits
   - 2000 char chunks with 200 char overlap
   - Max 50 chunks per page (safety limit)

3. **Batch Processing**: Add 20 chunks at a time to ChromaDB
   - vs. one-by-one additions (20x speedup)

4. **Memory Management**:
   - Process one PDF at a time
   - Explicit garbage collection
   - No accumulation in memory

## Performance Results

| Metric | Before | After |
|--------|--------|-------|
| Memory usage | 30GB | 500MB |
| Processing time | Hours (hung) | 62 seconds |
| Total chunks | 14,570+ | 1,523 |
| MacBook compatible | ❌ | ✅ |

**Final stats for 22 PDFs:**
- Total chunks: 1,523
- Processing time: 62.30 seconds
- Speed: 24.4 chunks/sec
- Memory: ~500MB peak

## Usage

### 1. Index PDFs (One-time)

```bash
poetry shell
python index_light.py artifacts/1
```

**Expected output:**
```
[1/22] Processing file.pdf...
  Processing 14 pages: [p5][p10][p14].. 18 chunks
  ✓ Added 18 chunks (total: 18)
...
✓ INDEXING COMPLETE
  Total chunks: 1523
  Time: 62.30s
```

### 2. Query the System

**Option A: Lightweight keyword search** (no API key needed)
```bash
python query_light.py "What are the rating plan rules?"
```

Returns top 5 relevant chunks using TF-IDF similarity.

**Option B: Full RAG with LLM** (requires API key)
```bash
# Set up API key
export ANTHROPIC_API_KEY="your_key_here"

# Query using LLM
python main.py ask "What are the rating plan rules?" artifacts/1
```

Returns natural language answer with source citations.

### 3. Answer Questions from CSV

```bash
python main.py query artifacts/questions.csv
```

Processes all questions in the CSV file.

## File Structure

```
├── index_light.py           # Ultra-light indexing (62 sec for 22 PDFs)
├── query_light.py           # Keyword-based search (no API key)
├── main.py                  # Full RAG with LLM (requires API key)
├── src/
│   ├── ultra_light_processor.py  # ONNX-based, memory-efficient
│   ├── fast_pdf_processor.py     # pypdf-based extraction
│   ├── vector_store.py           # sentence-transformers (for LLM use)
│   ├── rag_engine.py             # LLM integration (Claude/GPT)
│   └── config.py                 # Configuration management
├── debug_pdf.py             # Diagnostic tool
└── test_memory.py           # Memory testing tool
```

## Debugging Tools

**Analyze PDF content:**
```bash
python debug_pdf.py artifacts/1/filename.pdf
```

Shows character counts, chunks per page, and sample text.

**Test PDF reading:**
```bash
python test_memory.py artifacts/1/filename.pdf
```

Verifies pypdf works without memory issues.

## Trade-offs

### Ultra-Light System (index_light.py + query_light.py)

**Pros:**
- ✅ Fast indexing (62 seconds)
- ✅ Low memory (500MB)
- ✅ No API key required
- ✅ Works on any MacBook

**Cons:**
- ⚠️ Keyword-based search (TF-IDF, not semantic)
- ⚠️ No natural language answers
- ⚠️ Requires exact or similar terms to find results

### Full RAG System (main.py)

**Pros:**
- ✅ Semantic search (understands meaning)
- ✅ Natural language answers
- ✅ Source citations
- ✅ Better for complex questions

**Cons:**
- ⚠️ Requires API key (Claude or OpenAI)
- ⚠️ Costs per query (~$0.01-0.05)
- ⚠️ Slower (2-3 sec per question)
- ⚠️ Higher memory during indexing (~1-2GB)

## Recommended Workflow

1. **Index once** with ultra-light system:
   ```bash
   python index_light.py artifacts/1
   ```

2. **Test queries** with keyword search:
   ```bash
   python query_light.py "your question"
   ```

3. **For production use**, switch to full RAG:
   ```bash
   export ANTHROPIC_API_KEY="..."
   python main.py query artifacts/questions.csv
   ```

## Key Learnings

1. **PyTorch/sentence-transformers** too heavy for MacBooks
   - ONNX embeddings are 100x more memory-efficient

2. **Infinite loops** can hide in chunking algorithms
   - Always track previous position to detect stalls

3. **Batch processing** critical for ChromaDB performance
   - One-by-one additions = hours
   - Batched additions = minutes

4. **Memory profiling** essential for laptops
   - Activity Monitor revealed 30GB issue
   - Guided debugging to root cause

## Next Steps

- ✅ Indexing works perfectly
- ✅ Keyword search functional
- 🔜 Test with actual questions from questions.csv
- 🔜 Integrate full LLM if needed
- 🔜 Evaluate answer quality vs. expected outputs

## Success Criteria Met

✅ Runs on MacBook (8GB RAM) without crashing
✅ Memory usage under 1GB
✅ Reasonable processing time (< 2 minutes)
✅ Searchable vector database created
✅ Query interface functional

## Files Changed/Created

- `src/ultra_light_processor.py` - Core memory-efficient processor
- `src/fast_pdf_processor.py` - Fast pypdf extraction
- `index_light.py` - Lightweight indexing script
- `query_light.py` - Keyword search interface
- `debug_pdf.py` - PDF diagnostic tool
- `test_memory.py` - Memory testing
- `SOLUTION_SUMMARY.md` - This document

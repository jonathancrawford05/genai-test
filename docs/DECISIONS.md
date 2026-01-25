

# Design Decisions Log

This document explains key design decisions made during development, with rationale and tradeoffs.

---

## Architecture Decisions

### Decision 1: Multi-Agent Architecture (Router → Planner → Retriever → Orchestrator)

**Choice:** Decomposed RAG pipeline into specialized agents

**Rationale:**
- **Complex questions require multi-step reasoning** - EF_2 needs to find base rate, factor, then calculate
- **Document selection is critical** - With 22 PDFs, semantic search alone isn't enough
- **Modularity enables experimentation** - Can swap/tune individual agents independently
- **Mirrors real-world RAG systems** - Reflects industry patterns (e.g., LangGraph, LlamaIndex agents)

**Alternatives considered:**
- **Single-step RAG** - Simpler but can't handle multi-hop questions
- **ReACT agent** - More flexible but harder to debug, less predictable

**Tradeoff:**
- ✅ Better accuracy on complex questions
- ✅ Easier to debug (clear step boundaries)
- ❌ Higher latency (multiple LLM calls)
- ❌ More complex implementation

---

### Decision 2: Pre-Filtered Search (ChromaDB where clause)

**Choice:** Filter documents BEFORE semantic search, not after

**Rationale:**
- **Solved critical bug** - Search-then-filter failed for rate tables (returned prose about rates, not actual tables)
- **More efficient** - Only searches target documents, not all 1500+ chunks
- **Better precision** - No cross-document interference

**Alternatives considered:**
- **Search-then-filter** (original approach) - Failed on tabular data
- **Separate indexes per document** - Too complex, high memory overhead

**Tradeoff:**
- ✅ Finds chunks in rate tables (critical for EF_2)
- ✅ Faster retrieval
- ❌ Requires exact document filenames from Router
- ❌ Less serendipitous discovery

**Implementation:**
```python
# Pre-filtered
results = collection.query(
    query_texts=[query],
    where={"source_file": {"$in": target_documents}},
    n_results=top_k
)
```

---

### Decision 3: Tool Calling for Router (with fallback)

**Choice:** Two-step Router - reasoning + tool-validated selection

**Rationale:**
- **LLMs hallucinate document IDs** - llama3.2 returned `(214933333-...)` instead of `(214933334-...)`
- **Enum constraint prevents hallucination** - Tool forces selection from valid list
- **Preserves reasoning quality** - Step 1 allows free-form analysis

**Alternatives considered:**
- **Direct JSON output** - Simple but hallucination-prone
- **Post-hoc fuzzy matching** - Guesses LLM intent, less reliable
- **Constrained generation** - Not supported by Ollama

**Tradeoff:**
- ✅ Guaranteed valid filenames
- ✅ Maintains reasoning quality
- ❌ Requires 2 LLM calls (slower)
- ❌ Tool calling not always reliable (fallback needed)

---

### Decision 4: ONNX Embeddings (Standardized)

**Choice:** Use ONNX embeddings exclusively (all-MiniLM-L6-v2)

**Rationale:**
- **Performance superior to alternatives** - Experiments showed ONNX outperformed Ollama embeddings
- **Simpler codebase** - Single embedding model reduces complexity
- **Faster and smaller** - 79MB model, 384 dimensions, 2x faster than Ollama (768 dims)
- **No additional dependencies** - Built into ChromaDB, no Ollama server required for embeddings

**Previous approach:**
Originally supported both ONNX and Ollama embeddings (nomic-embed-text) for comparison. After experimentation, baseline ONNX configuration performed best, so Ollama embeddings were removed to simplify the system.

**Tradeoff:**
- ✅ Faster indexing and retrieval
- ✅ Simpler codebase (removed ~350 lines)
- ✅ Single directory structure (./chroma_db)
- ❌ Less flexibility for embedding experimentation
- ❌ Locked into 384-dimensional embeddings

**Note:** Ollama is still used for LLM inference (llama3.2), just not for embeddings.

---

## Implementation Decisions

### Decision 5: Chunk Size (1000 chars) and Overlap (200 chars)

**Choice:** 1000 character chunks with 200 character overlap

**Rationale:**
- **Insurance documents have long paragraphs** - 500 chars too small, splits concepts
- **Tables need context** - 1000 chars captures table + surrounding text
- **Overlap prevents information loss** - 200 chars ensures boundary concepts preserved

**Alternatives considered:**
- **Smaller chunks (500)** - Higher precision, but splits tables/lists
- **Larger chunks (2000)** - More context, but lower retrieval precision
- **Semantic chunking** - Better boundaries, but more complex

**Tradeoff:**
- ✅ Good balance for mixed content (prose + tables)
- ✅ Fast enough (~1500 chunks total)
- ❌ Not optimized per document type

---

### Decision 5a: Full-Document Chunking (Cross-Page Chunks)

**Choice:** Extract entire document text first, then chunk (chunks can span pages)

**Rationale:**
- **Critical bug discovered** - Rating rules table spans pages 3-4, page-by-page chunking only captured 22/35 rules
- **Many tables span pages** - TOC tables, rate tables, coverage tables in insurance PDFs
- **Simple solution** - Minor refactor (~60 lines) vs complex table detection
- **Page tracking preserved** - Chunks store page ranges (e.g., "3-4") instead of single pages

**Previous approach:**
Processed PDFs page-by-page, chunking each page independently. This prevented chunks from spanning page boundaries, causing multi-page tables to be split.

**Implementation:**
```python
# 1. Extract full document with page boundary tracking
full_text, page_boundaries = self._extract_full_document(reader)
# page_boundaries: [{page: 1, start_char: 0, end_char: 2341}, ...]

# 2. Chunk full text (crosses page boundaries naturally)
for chunk in self._chunk_text(full_text):
    chunk_start, chunk_end = get_position(chunk)
    page_range = self._determine_page_range(chunk_start, chunk_end, page_boundaries)
    # page_range could be "3" or "3-4" if chunk spans pages
```

**Tradeoff:**
- ✅ **Solves multi-page table problem** - 35/35 rules now captured
- ✅ **Better context preservation** - Natural content boundaries respected
- ✅ **Simple implementation** - No external dependencies
- ✅ **More informative metadata** - Page ranges show multi-page content
- ❌ Page numbers become ranges (acceptable - more informative actually)
- ❌ Slightly more complex position tracking (~40 lines)

**Impact:** This change was critical for achieving correct results on EF_1 (list all rating plan rules).

---

### Decision 6: Temperature = 0.0 for all agents

**Choice:** Deterministic LLM calls (temperature 0)

**Rationale:**
- **Reproducibility required for experiments** - Need consistent results across runs
- **Accuracy over creativity** - Insurance questions have definitive answers
- **Debugging easier** - Same input = same output

**Alternatives considered:**
- **Temperature 0.1-0.3** - Slight variation, may help with edge cases
- **Temperature 0.5+** - More creative, but inconsistent

**Tradeoff:**
- ✅ Reproducible experiments
- ✅ Predictable behavior
- ❌ May get "stuck" on wrong interpretation
- ❌ Less robust to prompt variations

---

### Decision 7: Document Summarization Pre-processing

**Choice:** Pre-generate summaries, not on-demand

**Rationale:**
- **Router needs summaries to select documents** - Can't route without knowing what's in each doc
- **Expensive operation** - 10-15 min for 22 docs, don't repeat per question
- **Summary quality is critical** - Manual review and refinement possible

**Alternatives considered:**
- **On-demand summarization** - Simple but slow, no caching
- **No summaries, embed titles only** - Fast but low-quality routing

**Tradeoff:**
- ✅ Fast routing (seconds not minutes)
- ✅ Reviewable summaries
- ❌ Requires setup step
- ❌ Summaries may be stale if PDFs change

---

## Experiment Design Decisions

### Decision 8: 4 Variations Testing Retrieval Parameters

**Choice:** Test 4 meaningful variations focusing on retrieval depth and chunk size

**Rationale:**
- **Retrieval depth matters** - Conservative vs high-depth tests precision/recall tradeoff
- **Chunk size affects context** - 1000 vs 2000 chars tests granularity vs context
- **Baseline establishes floor** - Need reference point for comparison
- **Focus on performance-impacting variables** - Test parameters that meaningfully affect results

**Variations:**
1. **Baseline** - Standard settings (top_k_docs=3, top_k_per_step=5, chunk_size=1000)
2. **High depth** - Test recall improvement (top_k_docs=5, top_k_per_step=10)
3. **Conservative** - Test speed improvement (top_k_docs=2, top_k_per_step=3)
4. **Large chunks** - Test context improvement (chunk_size=2000)

**Previous approach:**
Originally included "Ollama embeddings" variation, but baseline ONNX performed better and Ollama added complexity, so it was removed.

**Tradeoff:**
- ✅ Comprehensive comparison
- ✅ Tests multiple hypotheses
- ❌ 4x longer experiment time
- ❌ More results to analyze

---

### Decision 9: Partial Match Scoring

**Choice:** Custom metric based on keyword overlap, not just exact match

**Rationale:**
- **Exact match too strict** - Answer may be correct but phrased differently
- **EF_1 is a list** - Can count how many of 34 items are present
- **EF_2 is a number** - Can check if $604 appears in answer

**Implementation:**
```python
# For lists: Count matching items
expected_items = ["Rule A", "Rule B", ...]
matches = sum(1 for item in expected_items if item in actual)
score = matches / len(expected_items)

# For numbers: Check if present
if "$604" in actual or "604" in actual:
    score = 1.0
```

**Alternatives considered:**
- **Semantic similarity** - Embed both, compute cosine similarity (more robust but requires embeddings)
- **LLM-as-judge** - Ask LLM to evaluate (accurate but expensive)

**Tradeoff:**
- ✅ Fast and deterministic
- ✅ Works for both list and numeric questions
- ❌ Doesn't catch semantic equivalence
- ❌ Sensitive to exact phrasing

---

## Prompt Engineering Decisions

### Decision 10: Direct, Imperative Prompts

**Choice:** "Based on the retrieved information below, answer the following question:"

**Rationale:**
- **LLM was confused by indirect instructions** - Original prompt said "Instructions: 1. Answer the question..." which LLM interpreted as meta-instructions
- **Explicit repetition helps** - Repeating the question at the end ("Now, please answer the question '{question}'...") ensures LLM knows task

**Failed approach:**
```python
# This didn't work - too indirect
prompt = f"""
Question: {question}
Instructions:
1. Answer the question using ONLY the information above
...
Answer:
"""
```

**Working approach:**
```python
# This works - direct and explicit
prompt = f"""
Based on the retrieved information below, answer the following question:
QUESTION: {question}
...
Now, please answer the question "{question}" based ONLY on the information above.
ANSWER:
"""
```

**Tradeoff:**
- ✅ LLM understands task immediately
- ✅ Clear expectation setting
- ❌ Slightly more verbose
- ❌ May not transfer to other LLMs

---

## Rejected Approaches

### Rejected 1: Single-Pass RAG

**Why rejected:** Multi-hop questions like EF_2 require finding information in multiple documents, then combining. Single-pass can't structure this.

### Rejected 2: Embedding-Based Router

**Why rejected:** Embedding question + embedding doc summaries works, but LLM-based routing provides better reasoning (can consider doc types, not just semantic similarity).

### Rejected 3: Manual Index Management

**Why rejected:** Tracking chunk_size/overlap per index is good practice, but adds complexity. For prototype, acceptable to manually clean indexes when changing parameters. (See `future_refinements.md` for future implementation.)

### Rejected 4: Async/Parallel Retrieval

**Why rejected:** Retrieval steps often depend on each other (multi-hop). Parallelizing independent steps is premature optimization for prototype.

---

## Future Decisions Needed

### Open Question 1: How to handle missing information?

**Current:** LLM states "information is insufficient"

**Alternatives:**
- Return confidence score with answer
- Attempt web search for missing info
- Query user for clarification

### Open Question 2: Should we re-rank retrieved chunks?

**Current:** Trust ChromaDB's cosine similarity ranking

**Alternatives:**
- Cross-encoder re-ranking (more accurate, slower)
- LLM-based re-ranking (most accurate, expensive)
- Reciprocal Rank Fusion for multi-query

### Open Question 3: Optimal top_k values?

**Current:** top_k_docs=3, top_k_per_step=5

**Need experiments to determine:**
- Does increasing top_k improve accuracy? (likely yes)
- What's the point of diminishing returns? (test 1, 2, 3, 5, 10)
- Does it depend on question type?

---

## Lessons Learned

1. **Test with real questions early** - Academic examples don't reveal issues like table extraction
2. **LLMs are bad at exact numbers** - Need tool calling or validation for document IDs
3. **Search-then-filter fails on sparse text** - Pre-filtering is necessary for tables/structured data
4. **Prompt engineering matters enormously** - Small wording changes dramatically affect results
5. **Modular architecture pays off** - Could isolate and fix bugs in individual agents

---

## References

- ChromaDB Documentation: https://docs.trychroma.com
- Ollama Tool Calling: https://ollama.com/blog/tool-support
- Multi-Agent RAG Patterns: LlamaIndex, LangGraph documentation

---

**Last updated:** 2026-01-25

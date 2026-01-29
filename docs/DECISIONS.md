

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

### Decision 5b: Sliding Window Context Expansion

**Choice:** Retrieve with small chunks, expand with neighboring chunks for context

**Rationale:**
- **Semantic dilution problem discovered** - Large chunks (3000 chars) containing diverse content produce poor embeddings
- **Precision vs context tradeoff** - Small chunks (1000 chars) for precise retrieval, but may lack complete context
- **Best of both worlds** - Retrieve using small chunks, then expand with ±N neighbors to provide complete context
- **Configurable window** - Default ±2 chunks balances context and noise

**Problem solved:**
Originally tried increasing chunk_size to 3000 to capture full tables. This worked for preservation but retrieval got worse - chunks containing multiple diverse sections (rules + explanations + other content) produced diluted embeddings with lower similarity scores.

**Implementation:**
```python
# 1. Retrieve top-k chunks with small size (1000 chars) for precision
top_chunks = semantic_search(query, top_k=5)

# 2. Expand each chunk with neighbors (±2 chunks)
for chunk in top_chunks:
    neighbors = get_chunks_by_index(
        source_file=chunk.source_file,
        chunk_index_range=(chunk.index - 2, chunk.index + 2)
    )
    expanded_chunks.extend(neighbors)

# 3. Deduplicate and sort by original chunk index
# 4. Provide expanded context to LLM
```

**Tradeoff:**
- ✅ **Maintains retrieval precision** - Small chunks for accurate similarity matching
- ✅ **Provides complete context** - Neighboring chunks fill in missing information
- ✅ **Configurable** - Can adjust window size (0 = no expansion)
- ✅ **Preserves document order** - Chunks sorted by index, not similarity
- ❌ More chunks sent to LLM (3-5x more context)
- ❌ Requires global chunk indexing across documents

**Impact:** Critical for maintaining both retrieval precision and context completeness.

---

### Decision 5c: Configurable Chunking Strategy (Page vs Document)

**Choice:** Support both page-level and document-level chunking as configurable option

**Rationale:**
- **Different strategies have different tradeoffs** - Document-level preserves tables, page-level provides semantic focus
- **Use case dependent** - Enumeration questions benefit from document-level, conceptual questions from page-level
- **Experimentation needed** - Need to test both approaches to determine optimal strategy
- **Simple parameter** - One flag controls entire behavior

**Strategies:**

**Document-Level Chunking** (chunking_strategy="document"):
- Extracts full document first, then chunks (chunks can span pages)
- Preserves multi-page tables and lists
- Better for enumeration questions ("list all rules")
- Page numbers stored as ranges (e.g., "3-4")

**Page-Level Chunking** (chunking_strategy="page"):
- Chunks each page independently
- Chunks cannot span pages (may split tables)
- Better semantic focus (chunks don't mix topics)
- Better for conceptual questions
- Page numbers always single values (e.g., "3")

**Implementation:**
```python
# In BasePDFProcessor.__init__
self.chunking_strategy = chunking_strategy  # "page" or "document"

# In _process_single_pdf
if self.chunking_strategy == "page":
    return self._process_single_pdf_page_level(pdf_path)
else:  # "document"
    return self._process_single_pdf_document_level(pdf_path)
```

**Tradeoff:**
- ✅ **Flexibility** - Can choose best strategy per use case
- ✅ **Experimentation** - Easy to compare both approaches
- ✅ **Minimal code duplication** - Shared chunking logic
- ❌ Two code paths to maintain
- ❌ Metadata format differs (single page vs range)

**Experiment variations:**
- "sliding_window": Document-level + expansion
- "page_window": Page-level + expansion

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

### Decision 8: 6 Variations Testing Retrieval Strategies

**Choice:** Test 6 meaningful variations across multiple dimensions

**Rationale:**
- **Retrieval depth matters** - Conservative vs high-depth tests precision/recall tradeoff
- **Chunking strategy matters** - Page vs document level affects table preservation and semantic focus
- **Context expansion matters** - Sliding window tests precision vs context tradeoff
- **Search strategy matters** - Hybrid BM25+semantic vs pure semantic
- **Baseline establishes floor** - Need reference point for comparison

**Variations:**
1. **Baseline** - Document-level, no expansion, pure semantic (top_k=5)
2. **High Depth** - Document-level, no expansion, more retrieval (top_k=10)
3. **Conservative** - Document-level, no expansion, less retrieval (top_k=3)
4. **Sliding Window** - Document-level + ±2 chunk expansion (top_k=5)
5. **Page Window** - Page-level + ±2 chunk expansion (top_k=5)
6. **Hybrid Search** - Page-level + expansion + BM25+semantic hybrid

**Evolution:**
- **Version 1:** 4 variations testing retrieval depth and chunk size
- **Version 2:** Removed Ollama embeddings (ONNX performed better)
- **Version 3:** Added sliding window and page-level variations
- **Version 4 (current):** Added hybrid search variation

**Test Matrix:**
```
Variation       | Chunking  | Expansion | Search  | Top-K
----------------|-----------|-----------|---------|-------
baseline        | document  | 0         | semantic| 5
high_depth      | document  | 0         | semantic| 10
conservative    | document  | 0         | semantic| 3
sliding_window  | document  | ±2        | semantic| 5
page_window     | page      | ±2        | semantic| 5
hybrid_search   | page      | ±2        | hybrid  | 5
```

**Tradeoff:**
- ✅ Comprehensive comparison across 4 dimensions
- ✅ Tests complementary strategies
- ✅ Identifies optimal configuration
- ❌ 6x longer experiment time (~20-30 min)
- ❌ More complex result analysis

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

### Decision 11: Enumeration-Aware Planning

**Choice:** Add special handling for list/enumeration questions in Planner agent

**Rationale:**
- **Planner creating poor strategies** - For "List all rating plan rules", planner created overly specific multi-step plans
- **Wrong granularity** - Plans like "Step 1: Underwriting Guidelines, Step 2: Coverage limits" missed the actual rules
- **Query too specific** - Specific sub-queries retrieved 0 chunks, missing broad enumeration content
- **Simple questions need simple plans** - "List all X" should use single broad retrieval, not multi-step decomposition

**Problem:**
```python
# BAD PLAN - Too specific, misses items
{
  "strategy": "Break down into rule categories",
  "steps": [
    {"query": "Underwriting Guidelines rating rules", ...},  # 0 chunks
    {"query": "Coverage limits rating rules", ...},          # 0 chunks
  ],
  "requires_combination": true
}
```

**Solution:**
```python
# GOOD PLAN - Broad retrieval for enumeration
{
  "strategy": "Retrieve complete list from table of contents",
  "steps": [{
    "query": "table of contents section C rating plan rules complete list",
    "expected_output": "Complete enumeration of all rating plan rules"
  }],
  "requires_combination": false
}
```

**Implementation:**
```python
# 1. Detect enumeration patterns in question
enumeration_patterns = [
    "list all", "list the", "what are all", "enumerate",
    "show all", "give me all", "what are the"
]
is_enumeration = any(pattern in question.lower() for pattern in enumeration_patterns)

# 2. Add guidance to planner prompt
if is_enumeration:
    prompt += """
⚠️  ENUMERATION TASK DETECTED
- Use single-step retrieval focused on table of contents, index, or comprehensive lists
- Query should be broad: "table of contents [topic]", "complete list of [items]"
- Do NOT break into specific sub-categories (will miss items)
- Set requires_combination: false
"""

# 3. Provide correct/wrong examples in system prompt
```

**Tradeoff:**
- ✅ **Better strategies for list questions** - Single broad retrieval instead of multiple narrow ones
- ✅ **Targets right content** - Focuses on TOC, indexes, comprehensive lists
- ✅ **Prevents missing items** - Broad queries capture complete enumerations
- ✅ **Simpler plans** - Single-step is faster and less error-prone
- ❌ Pattern matching may miss edge cases
- ❌ More complex prompt engineering

**Impact:** Significantly improved retrieval for enumeration questions like "List all rating plan rules".

---

### Decision 12: Hybrid BM25 + Semantic Search

**Choice:** Combine BM25 keyword search with semantic embeddings using Reciprocal Rank Fusion

**Rationale:**
- **Pure semantic search insufficient** - Even with improved planning, retrieval poor on keyword-heavy queries
- **Enumeration questions keyword-heavy** - "list all rules" needs exact term matching ("C-1", "C-2", etc.)
- **Complementary strengths** - BM25 excellent for exact terms, semantic good for concepts
- **Adaptive weighting** - Different query types benefit from different search strategies

**Problem:**
Semantic search struggles with:
- Exact identifiers (rule numbers like "C-1", "C-35")
- Keyword-based enumerations (lists of specific items)
- Table of contents queries (need exact section names)

**Solution - Hybrid Retriever:**

1. **Build BM25 index** from all chunks in ChromaDB
2. **Run both searches** - semantic via ChromaDB, BM25 via rank-bm25
3. **Merge with RRF** (Reciprocal Rank Fusion):
   ```
   rrf_score = alpha * (1/(k + semantic_rank)) + (1-alpha) * (1/(k + bm25_rank))
   ```
4. **Adaptive alpha** based on query type:
   - Enumeration queries: alpha=0.3 (70% BM25, 30% semantic)
   - Reasoning queries: alpha=0.7 (30% BM25, 70% semantic)

**Implementation:**
```python
# In retriever_agent.py
if self.config.use_hybrid and self.hybrid_retriever:
    # Detect query type
    is_enumeration = any(term in query_lower for term in [
        "list", "all", "table of contents", "enumerate", "rules"
    ])

    # Set adaptive alpha
    alpha = 0.3 if is_enumeration else 0.7

    # Hybrid search
    results = self.hybrid_retriever.search(
        query=query,
        top_k=top_k,
        alpha=alpha,  # BM25-heavy for enumerations, semantic-heavy for reasoning
        where=where_clause
    )
```

**BM25 Tokenization:**
```python
def _tokenize(self, text: str) -> List[str]:
    text = text.lower()
    # Keep alphanumeric + hyphens (preserves "C-1", "C-35")
    tokens = re.findall(r'\b[\w\-]+\b', text)
    return tokens
```

**Tradeoff:**
- ✅ **Better keyword matching** - Finds exact terms that semantic search misses
- ✅ **Robust to query type** - Adapts weighting based on task
- ✅ **Complementary strengths** - Gets best of both approaches
- ✅ **Proven technique** - RRF is well-established in IR
- ❌ Additional index to maintain (BM25)
- ❌ Slower retrieval (2 searches + merge)
- ❌ More complex implementation (~300 lines)

**Performance:**
- BM25 index build: ~5-10 seconds for 1500 chunks
- Hybrid search: ~2x semantic search time (still fast at <1 second)

**Impact:** Critical improvement for enumeration questions, maintains performance on reasoning questions.

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

### ~~Open Question 2: Should we re-rank retrieved chunks?~~ ✅ RESOLVED

**Resolution:** Implemented hybrid BM25 + semantic search with RRF (Reciprocal Rank Fusion)

**Result:** RRF provides effective re-ranking by merging BM25 and semantic rankings. Cross-encoder re-ranking could still improve further but adds significant latency.

### ~~Open Question 3: Optimal top_k values?~~ ✅ PARTIALLY RESOLVED

**Current experiments test:** top_k = 3, 5, 10

**Findings from experiments will determine:**
- Does increasing top_k improve accuracy? (testing in progress)
- What's the point of diminishing returns?
- Does it depend on question type? (likely yes based on enumeration vs reasoning)

### Open Question 4: Optimal sliding window size?

**Current:** ±2 chunks (provides ~5000 chars of context)

**Need experiments to determine:**
- Does larger window improve accuracy?
- At what point does noise outweigh signal?
- Should window size be adaptive based on chunk similarity?

### Open Question 5: Optimal hybrid alpha values?

**Current:**
- Enumeration: alpha=0.3 (70% BM25, 30% semantic)
- Reasoning: alpha=0.7 (30% BM25, 70% semantic)

**Need experiments to determine:**
- Are these ratios optimal?
- Should alpha be continuous based on query features?
- Can we learn optimal alpha from training data?

---

## Lessons Learned

1. **Test with real questions early** - Academic examples don't reveal issues like table extraction
2. **LLMs are bad at exact numbers** - Need tool calling or validation for document IDs
3. **Search-then-filter fails on sparse text** - Pre-filtering is necessary for tables/structured data
4. **Prompt engineering matters enormously** - Small wording changes dramatically affect results
5. **Modular architecture pays off** - Could isolate and fix bugs in individual agents
6. **Multi-page content is common** - Tables, lists, and sections frequently span pages in real documents
7. **Semantic dilution is real** - Large chunks with diverse content produce poor embeddings
8. **Sliding window is powerful** - Retrieve with precision, expand for context - best of both worlds
9. **Query type matters** - Enumeration vs reasoning queries need different retrieval strategies
10. **Pure semantic search insufficient** - Hybrid BM25+semantic necessary for robust retrieval
11. **Adaptive strategies win** - Systems that adjust behavior based on query type outperform fixed approaches
12. **Simplification improves performance** - Removing Ollama embeddings reduced complexity and improved results

---

## References

- ChromaDB Documentation: https://docs.trychroma.com
- Ollama Tool Calling: https://ollama.com/blog/tool-support
- Multi-Agent RAG Patterns: LlamaIndex, LangGraph documentation

---

## System Evolution Summary

### Phase 1: Initial Multi-Agent Implementation
- Router → Planner → Retriever → Orchestrator pipeline
- Both ONNX and Ollama embeddings supported
- Page-by-page chunking
- Pure semantic search

### Phase 2: Simplification and Bug Fixes
- **Removed Ollama embeddings** - ONNX outperformed, simplified codebase
- **Fixed multi-page table splitting** - Implemented full-document chunking
- Single directory structure (./chroma_db)

### Phase 3: Context Optimization
- **Discovered semantic dilution** - Large chunks produce poor embeddings
- **Implemented sliding window** - Retrieve with small chunks, expand for context
- **Configurable chunking** - Support both page-level and document-level

### Phase 4: Query-Aware Retrieval
- **Enumeration-aware planning** - Special handling for "list all" questions
- **Hybrid BM25 + semantic search** - Combines keyword and semantic ranking
- **Adaptive weighting** - Adjusts search strategy based on query type

### Current State (6 Variations)
- Baseline (document, no expansion, semantic)
- High Depth (more retrieval)
- Conservative (less retrieval)
- Sliding Window (document + expansion)
- Page Window (page + expansion)
- Hybrid Search (page + expansion + BM25)

---

**Last updated:** 2026-01-26

# Future Refinements & Improvements

This document tracks potential improvements, optimizations, and features for future development. Items are organized by priority and component.

---

## High Priority

### 1. Index Parameter Tracking & Management

**Problem:**
Currently, ChromaDB indexes are created without tracking the parameters used (chunk_size, chunk_overlap, embedding model). If you change these parameters in code, the system continues using the old index without warning, leading to inconsistent results.

**Current workaround:**
- Manually delete `./chroma_db_onnx/` or `./chroma_db_ollama/` directories
- Or use different persist_directory per parameter combination

**Proposed solution:**
```python
# Store index metadata in a separate config file
# Example: ./chroma_db_onnx/index_config.json
{
    "created_at": "2025-01-24T10:30:00Z",
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "embedding_model": "all-MiniLM-L6-v2",
    "embedding_dimensions": 384,
    "pdf_count": 22,
    "total_chunks": 1523,
    "source_folder": "artifacts/1",
    "last_updated": "2025-01-24T10:30:00Z"
}

# Index validation on load
def validate_or_recreate_index(processor, expected_params):
    config_path = f"{processor.persist_directory}/index_config.json"

    if not config_path.exists():
        return recreate_index(processor)

    stored_params = load_config(config_path)

    # Check critical parameters
    if (stored_params['chunk_size'] != expected_params['chunk_size'] or
        stored_params['chunk_overlap'] != expected_params['chunk_overlap'] or
        stored_params['embedding_model'] != expected_params['embedding_model']):

        print(f"⚠️  Index parameters changed:")
        print(f"   Stored: {stored_params}")
        print(f"   Expected: {expected_params}")
        print(f"   Recreating index...")

        return recreate_index(processor)

    print(f"✓ Index valid: {stored_params['total_chunks']} chunks")
    return processor.collection
```

**Benefits:**
- Automatic index invalidation when parameters change
- Clear debugging info (when was index created, with what params)
- Supports A/B testing different chunking strategies
- Prevents silent bugs from parameter mismatches

**Implementation effort:** Medium (2-4 hours)

**Related files:**
- `src/base_processor.py` - Add config save/load methods
- `src/onnx_processor.py` - Update `_setup_collection()`
- `src/ollama_processor.py` - Update `_setup_collection()`

---

### 2. Configurable Chunk Parameters

**Problem:**
Chunk size (1000) and overlap (200) are hardcoded in base_processor.py:38-39. Experimenting with different chunking strategies requires code changes.

**Current state:**
```python
# base_processor.py
self.chunk_size = 1000  # Hardcoded
self.chunk_overlap = 200  # Hardcoded
```

**Proposed solution:**
```python
@dataclass
class ChunkingConfig:
    chunk_size: int = 1000
    chunk_overlap: int = 200
    chunking_strategy: str = "fixed"  # "fixed", "semantic", "sentence"
    min_chunk_size: int = 100
    max_chunk_size: int = 2000

class BasePDFProcessor:
    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        collection_name: str = "pdf_chunks",
        chunking_config: Optional[ChunkingConfig] = None
    ):
        self.chunking_config = chunking_config or ChunkingConfig()
        self.chunk_size = self.chunking_config.chunk_size
        self.chunk_overlap = self.chunking_config.chunk_overlap
        # ...
```

**Benefits:**
- Easy experimentation with chunk sizes
- Support for different chunking strategies in future
- Configuration passed explicitly, not hidden in code
- Works with index parameter tracking (#1 above)

**Implementation effort:** Low (1-2 hours)

**Experiment ideas:**
- Small chunks (500 chars): Better precision, more chunks
- Large chunks (2000 chars): More context, fewer chunks
- Semantic chunking: Split on paragraph boundaries
- Sentence-based chunking: Keep sentences intact

---

### 3. Pre-filtered Search Option

**Problem:**
Current approach searches entire index then filters by metadata. This retrieves many irrelevant chunks that are immediately discarded.

**Current approach:**
```python
# Search ALL documents
results = processor.query(query, top_k=10)

# Filter to target documents (may discard 8/10 chunks)
for chunk in results:
    if chunk.source_file not in target_documents:
        continue  # Wasted retrieval
```

**Proposed solution:**
Add optional pre-filtering using ChromaDB's `where` clause:

```python
class RetrieverConfig:
    use_prefiltering: bool = False  # Feature flag

def _execute_step(self, step, verbose=False):
    if self.config.use_prefiltering and step.target_documents:
        # Pre-filter search to target documents only
        results = self.processor.collection.query(
            query_texts=[step.query],
            n_results=self.config.top_k_per_step,
            where={"source_file": {"$in": step.target_documents}}
        )
    else:
        # Current approach: search all, filter after
        results = self.processor.query(step.query, self.config.top_k_per_step)
        # ... filter by metadata ...
```

**Tradeoffs:**
| Approach | Pros | Cons |
|----------|------|------|
| Search-then-filter (current) | Simple, finds cross-doc relevance | Retrieves irrelevant chunks |
| Pre-filtered search | More efficient, targeted results | Requires exact filenames, less serendipity |

**Recommendation:**
- Add as optional feature flag
- Run experiments comparing recall and precision
- May be useful for large document sets (100+ PDFs)

**Implementation effort:** Low (2-3 hours)

---

### 4. Router Confidence Scores

**Problem:**
Router selects top-k documents but provides no confidence scores. Unclear how confident it is in selections, especially when falling back to alphabetical order.

**Proposed solution:**
```python
@dataclass
class DocumentSelection:
    filename: str
    confidence: float  # 0.0-1.0
    reasoning: str

class RouterAgent:
    def select_documents(self, question, top_k=3) -> List[DocumentSelection]:
        # Updated prompt to request confidence scores
        prompt = """
        For each selected document, provide:
        1. Filename (exact)
        2. Relevance score (0.0-1.0)
        3. Brief reasoning

        Return JSON:
        [
            {
                "filename": "doc.pdf",
                "confidence": 0.95,
                "reasoning": "Contains rate tables for requested tier"
            },
            ...
        ]
        """
        # Parse structured response
        selections = parse_selections(response)
        return selections
```

**Benefits:**
- Filter low-confidence selections (e.g., confidence < 0.3)
- Adjust top_k dynamically based on confidence
- Better debugging ("Why was this doc selected?")
- Metrics for Router quality

**Implementation effort:** Medium (3-4 hours)

---

## Medium Priority

### 5. Semantic Chunking

**Problem:**
Fixed-size chunking can split mid-sentence or mid-paragraph, losing coherence.

**Example issue:**
```
Chunk 1: "...the property must meet all requirements. These include: 1) Location in approved zone 2) Build"
Chunk 2: "ing type must be single-family residential 3) No prior claims..."
```

**Proposed solution:**
Use LangChain's semantic chunking or sentence-based splitting:

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

class SemanticChunker:
    def __init__(self, chunk_size=1000, overlap=200):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ". ", " ", ""],  # Split on natural boundaries
            length_function=len,
            is_separator_regex=False,
        )

    def chunk_text(self, text):
        return self.splitter.split_text(text)
```

**Benefits:**
- Chunks maintain semantic coherence
- Better context for LLM
- Improved retrieval quality

**Implementation effort:** Medium (4-6 hours)

**Related:** Issue #2 (configurable chunking)

---

### 6. Multi-Vector Retrieval

**Problem:**
Single query per step may miss relevant content phrased differently.

**Example:**
- Query: "ineligible risk rules"
- Missed content: "properties that cannot be insured"

**Proposed solution:**
Generate multiple query variations per step:

```python
def _execute_step_multi_query(self, step):
    # Generate query variations
    queries = [
        step.query,  # Original
        self._rephrase_query(step.query, style="formal"),
        self._rephrase_query(step.query, style="conversational"),
        self._extract_keywords(step.query)
    ]

    all_chunks = []
    seen_ids = set()

    for query in queries:
        results = self.processor.query(query, top_k=5)
        for chunk in results:
            chunk_id = f"{chunk.source_file}:{chunk.chunk_index}"
            if chunk_id not in seen_ids:
                all_chunks.append(chunk)
                seen_ids.add(chunk_id)

    # Re-rank combined results
    return self._rerank_chunks(all_chunks, step.query)[:self.config.top_k_per_step]
```

**Benefits:**
- Higher recall (find more relevant chunks)
- Robust to query phrasing
- Better for ambiguous questions

**Tradeoffs:**
- 3-4x slower (multiple searches per step)
- More LLM calls (for query rephrasing)
- May retrieve noise

**Implementation effort:** Medium-High (6-8 hours)

---

### 7. Answer Citation & Verification

**Problem:**
Once Orchestrator generates answer (Phase 5), need to cite sources and verify accuracy.

**Proposed features:**
```python
@dataclass
class Answer:
    text: str
    citations: List[Citation]
    confidence: float
    verification_status: str  # "verified", "partial", "uncertain"

@dataclass
class Citation:
    text: str  # Cited text from answer
    source_file: str
    page_number: int
    chunk_text: str  # Original chunk content
    relevance_score: float

def generate_answer_with_citations(execution_result: ExecutionResult, question: str) -> Answer:
    # Format chunks with source IDs
    context = ""
    for i, step_result in enumerate(execution_result.step_results):
        for j, chunk in enumerate(step_result.chunks):
            source_id = f"[{i}-{j}]"
            context += f"{source_id} {chunk['text']} (Source: {chunk['source_file']}, p{chunk['page_number']})\n"

    # Prompt LLM to cite sources
    prompt = f"""
    Question: {question}

    Context:
    {context}

    Instructions:
    1. Answer the question using ONLY the provided context
    2. Cite sources using [i-j] notation
    3. If uncertain or info missing, state clearly

    Answer:
    """

    response = llm.generate(prompt)

    # Parse citations
    citations = extract_citations(response, execution_result)

    # Verify all claims are cited
    verification = verify_citations(response, citations)

    return Answer(
        text=response,
        citations=citations,
        confidence=calculate_confidence(citations),
        verification_status=verification
    )
```

**Benefits:**
- Transparency (show where answer came from)
- Verification (ensure claims are grounded)
- Debugging (identify missing info)

**Implementation effort:** High (8-12 hours)

**Critical for:** Production use, regulatory compliance

---

### 8. Hybrid Search (Dense + Sparse)

**Problem:**
Pure semantic search may miss exact keyword matches (e.g., "Tier 1", "PC 5", "$500k").

**Example:**
- Query: "premium for Tier 1 PC 5"
- Semantic search finds: "coverage amounts for highest tier property class"
- Misses: Exact "Tier 1" and "PC 5" mentions

**Proposed solution:**
Combine dense (semantic) and sparse (keyword/BM25) retrieval:

```python
class HybridRetriever:
    def __init__(self, processor, alpha=0.5):
        self.processor = processor  # Dense retrieval
        self.bm25 = BM25Retriever()  # Sparse retrieval
        self.alpha = alpha  # Weight: 0=pure sparse, 1=pure dense

    def retrieve(self, query, top_k=10):
        # Dense retrieval (semantic)
        dense_results = self.processor.query(query, top_k=top_k)

        # Sparse retrieval (keyword/BM25)
        sparse_results = self.bm25.query(query, top_k=top_k)

        # Combine with weighted scores
        combined = self._reciprocal_rank_fusion(
            dense_results,
            sparse_results,
            alpha=self.alpha
        )

        return combined[:top_k]
```

**Benefits:**
- Best of both: semantic understanding + exact matches
- Better for queries with specific terms/numbers
- Common in production RAG systems

**Tradeoffs:**
- More complex implementation
- Requires maintaining two indexes
- Tuning alpha parameter

**Implementation effort:** High (10-15 hours)

---

## Low Priority / Nice to Have

### 9. Caching Layer

Add LRU cache for repeated queries to reduce latency and cost.

```python
from functools import lru_cache

class CachedRetriever(RetrieverAgent):
    @lru_cache(maxsize=128)
    def _execute_step_cached(self, step_hash):
        return super()._execute_step(step)
```

**Implementation effort:** Low (1-2 hours)

---

### 10. Async/Parallel Execution

Execute retrieval steps in parallel when they don't depend on each other.

```python
async def execute_plan_async(self, plan):
    # Identify independent steps
    independent_steps = identify_independent(plan.steps)

    # Execute in parallel
    tasks = [self._execute_step_async(step) for step in independent_steps]
    results = await asyncio.gather(*tasks)
```

**Implementation effort:** Medium (4-6 hours)

---

### 11. Table Extraction & Specialized Handling

**Problem:**
Tables in PDFs (like rate tables) may be poorly extracted as text.

**Solution:**
- Use table detection (pdfplumber, camelot)
- Store tables as structured data
- Separate retrieval path for tabular queries

**Implementation effort:** High (15-20 hours)

---

### 12. Evaluation Framework

**Critical for Part 2 of test (experimentation harness).**

```python
@dataclass
class EvaluationMetrics:
    accuracy: float  # Exact match with expected answer
    relevance: float  # Retrieved chunks contain answer
    precision: float  # % relevant chunks / total retrieved
    recall: float  # % relevant chunks retrieved / total relevant
    latency: float  # Total time (ms)
    cost: float  # LLM API cost

class ExperimentRunner:
    def run_experiment(self, config, test_cases):
        results = []
        for question, expected_answer, pdf_folder in test_cases:
            start = time.time()

            answer = answer_pdf_question(question, pdf_folder, config)

            metrics = EvaluationMetrics(
                accuracy=compute_accuracy(answer, expected_answer),
                relevance=compute_relevance(answer, chunks),
                precision=compute_precision(chunks, expected_answer),
                recall=compute_recall(chunks, expected_answer),
                latency=(time.time() - start) * 1000,
                cost=calculate_cost(llm_calls)
            )

            results.append((question, answer, metrics))

        return ExperimentReport(config, results)
```

**Implementation effort:** High (12-16 hours)

---

## Experiments to Run

Once refinements are implemented, run these experiments:

### Experiment 1: Chunking Strategy
- **Variables:** chunk_size [500, 1000, 2000], chunk_overlap [0, 100, 200, 400]
- **Metric:** Accuracy on EF_1 and EF_2
- **Hypothesis:** Larger chunks provide more context but reduce precision

### Experiment 2: Embedding Model
- **Variables:** ONNX (all-MiniLM-L6-v2) vs Ollama (nomic-embed-text)
- **Metrics:** Accuracy, latency, memory
- **Hypothesis:** Ollama higher quality but 2x slower

### Experiment 3: Top-k Documents
- **Variables:** top_k [1, 2, 3, 5]
- **Metric:** Accuracy, latency
- **Hypothesis:** Diminishing returns after 3 documents

### Experiment 4: Retrieval Strategy
- **Variables:** Search-then-filter vs Pre-filtered vs Hybrid
- **Metrics:** Precision, recall, latency
- **Hypothesis:** Hybrid best quality, pre-filtered fastest

### Experiment 5: Multi-step vs Single-step
- **Variables:** Force single-step plan vs allow multi-step
- **Metric:** Accuracy on EF_2 (multi-hop question)
- **Hypothesis:** Multi-step significantly better for complex questions

---

## Implementation Priority

**Phase 5 (Current):**
- Implement Orchestrator agent (answer generation)

**Phase 6 (Next):**
- #12: Evaluation framework (required for Part 2)
- #1: Index parameter tracking (critical for experiments)
- #2: Configurable chunk parameters (enables experiments)

**Phase 7 (Polish):**
- #4: Router confidence scores
- #7: Answer citations
- #5: Semantic chunking

**Phase 8 (Optimization):**
- #3: Pre-filtered search (optional)
- #6: Multi-query retrieval
- #8: Hybrid search

**Phase 9 (Production):**
- #9: Caching
- #10: Async execution
- #11: Table handling

---

## Open Questions

1. **Should Planner see full document text or just summaries?**
   - Currently: Just summaries (via Router output)
   - Alternative: Give Planner sample chunks to create better queries

2. **Should Retriever re-rank results?**
   - Currently: Uses ChromaDB's cosine similarity ranking
   - Alternative: Re-rank with cross-encoder or LLM

3. **How to handle multi-document answers?**
   - Currently: Retriever combines chunks, Orchestrator synthesizes
   - Alternative: Orchestrator explicitly cites which document answered which part

4. **What's the right balance of top_k values?**
   - Router top_k: 3 documents
   - Retriever top_k: 10 chunks per step
   - Are these optimal? Need experiments.

---

## Notes

This is a living document. Add new refinements as you discover limitations or opportunities during development and testing.

Last updated: 2025-01-24

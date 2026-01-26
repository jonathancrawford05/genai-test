# Multi-Agent RAG System Architecture

## Overview

This system implements a multi-agent Retrieval-Augmented Generation (RAG) pipeline for complex PDF question answering. It uses a four-agent architecture: Router → Planner → Retriever → Orchestrator.

## Key Features (Current State)

1. **Hybrid BM25 + Semantic Search** - Combines keyword and semantic ranking with adaptive weighting
2. **Sliding Window Expansion** - Retrieves with small chunks, expands with neighbors for context
3. **Enumeration-Aware Planning** - Special handling for "list all" questions
4. **Full-Document Chunking** - Chunks can span pages to preserve multi-page tables
5. **Pre-Filtered Search** - Filters documents BEFORE semantic search for efficiency
6. **ONNX Embeddings Only** - Single embedding model for simplicity and performance

## System Components

### 1. Document Processing Layer

**Base Processor** (`src/base_processor.py`)
- Abstract base class for PDF processing
- Handles chunking with configurable size, overlap, and strategy
- Manages ChromaDB collections for vector storage
- Default: 1000 char chunks, 200 char overlap, document-level chunking

**Chunking Strategies:**
- **Document-level** (`chunking_strategy="document"`)
  - Extracts full document text, then chunks (chunks can span pages)
  - Preserves multi-page tables and lists
  - Page metadata stored as ranges (e.g., "3-4")
  - Better for enumeration questions
- **Page-level** (`chunking_strategy="page"`)
  - Chunks each page independently (chunks cannot span pages)
  - Better semantic focus, may split tables
  - Page metadata always single values (e.g., "3")
  - Better for conceptual questions

**Embedding Implementation:**
- **ONNXProcessor** (`src/onnx_processor.py`)
  - Model: all-MiniLM-L6-v2
  - Size: 79MB
  - Dimensions: 384
  - Runtime: CPU-optimized ONNX
  - Collection: `chroma_db/`
  - Chosen for performance after experimentation (Ollama embeddings removed)

**Hybrid Retriever** (`src/hybrid_retriever.py`)
- Combines BM25 keyword search with semantic embeddings
- Uses Reciprocal Rank Fusion (RRF) for score merging
- Adaptive alpha weighting based on query type:
  - Enumeration queries: 70% BM25, 30% semantic
  - Reasoning queries: 30% BM25, 70% semantic

### 2. Multi-Agent Pipeline

#### **Phase 1: Document Summarization**
- Pre-processing step that generates summaries for each PDF
- Output: `artifacts/document_summaries.json`
- Summary schema:
  ```json
  {
    "filename.pdf": {
      "document_type": "Rate Manual",
      "summary": "...",
      "key_topics": ["topic1", "topic2"],
      "use_for": "pricing, rating calculations"
    }
  }
  ```

#### **Phase 2: Router Agent** (`src/agents/router_agent.py`)

**Purpose:** Select top-k most relevant documents from summaries

**Input:**
- Question string
- Document summaries JSON
- top_k (default: 3)

**Process:**
1. Formats all document summaries for LLM
2. Calls LLM (llama3.2) with routing prompt
3. Parses JSON response to extract filenames
4. Validates filenames exist in summaries
5. Falls back to alphabetical selection if parsing fails

**Output:** List of exact document filenames
```python
["(215066178-180449588)-CT MAPS Homeowner Rules Manual eff 08.18.25 v4.pdf", ...]
```

**Key Features:**
- Temperature: 0.0 (deterministic)
- Emphasizes returning EXACT filenames including ID prefixes
- Validates against available summaries
- Provides warnings if documents not found

#### **Phase 3: Planner Agent** (`src/agents/planner_agent.py`)

**Purpose:** Create retrieval strategy optimized for question type

**Input:**
- Question string
- Selected documents from Router

**Process:**
1. **Detects query type:**
   - Enumeration patterns: "list all", "enumerate", "what are all"
   - Reasoning patterns: "calculate", "what is", "how does"
2. **Analyzes question complexity**
3. **Creates appropriate strategy:**
   - **Enumeration questions:** Single-step broad retrieval targeting TOC/indexes
   - **Reasoning questions:** Multi-step retrieval with synthesis
4. Creates retrieval plan with:
   - Overall strategy description
   - Numbered retrieval steps
   - Target documents per step
   - Specific queries per step
   - Expected output format

**Output:** RetrievalPlan object
```python
RetrievalPlan(
    question="What is the premium for Tier 1, PC 5, $500k, 2% deductible?",
    strategy="Multi-step retrieval: 1) Find tier definitions, 2) Locate rate tables, 3) Extract specific rate",
    steps=[
        RetrievalStep(
            step_number=1,
            description="Find Tier 1 definition and requirements",
            target_documents=["rate_manual.pdf"],
            query="Tier 1 property classification definition requirements",
            expected_output="Definition and criteria for Tier 1 properties"
        ),
        ...
    ],
    requires_combination=True
)
```

**Example Enumeration Plan:**
```python
# For "List all rating plan rules"
RetrievalPlan(
    question="List all rating plan rules",
    strategy="Single-step retrieval targeting table of contents",
    steps=[
        RetrievalStep(
            step_number=1,
            description="Retrieve complete list of rating plan rules",
            target_documents=["Homeowner Rules Manual.pdf"],
            query="table of contents section C rating plan rules complete list",
            expected_output="Complete enumeration of all rating plan rules"
        )
    ],
    requires_combination=False
)
```

**Key Features:**
- **Enumeration-aware:** Detects "list all" patterns, uses single-step broad retrieval
- **Query optimization:** Targets TOC, indexes, comprehensive lists for enumerations
- **Handles both simple (1-step) and complex (multi-step) questions**
- Assigns specific target documents per step
- Creates focused queries for each step
- Indicates if steps need to be combined for final answer

#### **Phase 4: Retriever Agent** (`src/agents/retriever_agent.py`)

**Purpose:** Execute retrieval plan with hybrid search and context expansion

**Input:**
- RetrievalPlan from Planner
- Initialized processor (ONNX)
- RetrieverConfig with expansion and hybrid settings

**Process:**
1. **Initialize hybrid retriever** (if enabled)
   - Builds BM25 index from all chunks in ChromaDB
2. **Execute each step in plan sequentially**
3. **For each step:**
   - **Detects query type** (enumeration vs reasoning)
   - **Hybrid search** (if enabled):
     - Runs semantic search via ChromaDB
     - Runs BM25 keyword search
     - Merges with RRF using adaptive alpha
     - Enumeration: 70% BM25, 30% semantic
     - Reasoning: 30% BM25, 70% semantic
   - **Pre-filtered search:**
     - ChromaDB where clause filters to target documents BEFORE search
     - More efficient, better precision on tables
   - **Sliding window expansion** (if enabled):
     - Retrieves neighboring chunks (±N) for each result
     - Provides complete context while maintaining retrieval precision
     - Deduplicates and sorts by chunk index
4. Combines results if `plan.requires_combination=True`

**Output:** ExecutionResult with all retrieved chunks
```python
ExecutionResult(
    plan=...,
    step_results=[
        RetrievalResult(
            step=...,
            chunks=[
                {
                    'text': 'chunk content...',
                    'source_file': 'document.pdf',
                    'page_number': 42,
                    'chunk_index': 15,
                    'distance': 0.234
                },
                ...
            ]
        ),
        ...
    ]
)
```

**Key Features:**
- **Hybrid BM25 + Semantic Search:** Combines keyword and semantic ranking
- **Adaptive weighting:** Adjusts alpha based on query type
- **Sliding window expansion:** Retrieves with precision, expands for context
- **Pre-filtered search:** ChromaDB where clause filtering before search
- **Auto-indexing:** Indexes PDFs if processor is empty
- **Verbose diagnostics:** Shows retrieval details, search types, chunk counts
- **Configurable:** expansion window, hybrid alpha, chunking strategy

## Retrieval Architecture: Pre-Filtered Search

### How It Works

**Question: Does the Retriever only search within selected documents, or search everything then filter?**

**Answer: Pre-filters to target documents BEFORE semantic search.**

```python
# Build where clause for ChromaDB
where_clause = None
if step.target_documents:
    where_clause = {"source_file": {"$in": step.target_documents}}

# Hybrid search (if enabled)
if self.config.use_hybrid and self.hybrid_retriever:
    # Detect query type for adaptive weighting
    is_enumeration = any(term in query.lower() for term in [
        "list", "all", "table of contents", "enumerate", "rules"
    ])

    # Set alpha: enumeration=0.3 (BM25-heavy), reasoning=0.7 (semantic-heavy)
    alpha = 0.3 if is_enumeration else 0.7

    # Hybrid BM25 + semantic with pre-filtering
    results = self.hybrid_retriever.search(
        query=step.query,
        top_k=self.config.top_k_per_step,
        alpha=alpha,
        where=where_clause,  # Pre-filter to target documents
        verbose=verbose
    )
else:
    # Pure semantic search with pre-filtering
    if where_clause:
        results = self.processor.collection.query(
            query_texts=[step.query],
            where=where_clause,  # Pre-filter
            n_results=self.config.top_k_per_step
        )
    else:
        results = self.processor.query(step.query, top_k=self.config.top_k_per_step)

# Sliding window expansion (if enabled)
if self.config.expand_context > 0:
    expanded_chunks = self._expand_chunks_with_window(chunks)
```

### Why This Approach?

**Pros:**
- **More efficient** - Only searches target documents, not entire index
- **Better precision** - Finds chunks in tables and structured data
- **Prevents cross-document interference** - No irrelevant chunks from other docs
- **Exact filename matching** - Requires Router to provide correct filenames
- **Supports hybrid search** - Pre-filtering works with both BM25 and semantic

**Cons:**
- Requires exact document filenames from Router
- No serendipitous cross-document discovery
- Slightly more complex where clause construction

**Previous approach (search-then-filter) was replaced because:**
- Failed to find chunks in rate tables (returned prose about rates instead)
- Inefficient for large document collections
- Cross-document interference reduced retrieval quality

## Index Management

### Current Approach

**Index Storage:**
- Single persistent ChromaDB directory: `./chroma_db/`
- Multiple collections possible (different chunk sizes, strategies)
- Collection names encode parameters: `pdf_{chunk_size}_{chunk_overlap}`

**Index Lifecycle:**
```python
# 1. Check if collection exists
collection = client.get_collection(name=collection_name)

# 2. Validate dimensions match expected (recreate if mismatch)
if stored_dims != expected_dims:
    client.delete_collection(name)
    collection = client.create_collection(name)

# 3. Check if collection has data
if collection.count() == 0:
    # Index PDFs from folder with configured strategy
    processor.process_folder(pdf_folder)
```

**Index Creation Parameters:**
- Chunk size: 1000 characters (configurable)
- Chunk overlap: 200 characters (configurable)
- Chunking strategy: "document" or "page" (configurable)
- Embedding model: ONNX (all-MiniLM-L6-v2, 384 dims)
- Metadata: source_file, page_number (or range), chunk_index, total_pages

**BM25 Index (Hybrid Search):**
- Built in-memory from ChromaDB chunks
- Tokenization: lowercase + alphanumeric + hyphens
- No persistence (rebuilt on each run, takes ~5-10 seconds)

### Limitations

**Problem:** No parameter tracking
- If you change chunk_size or overlap, old index is still used
- No way to know which parameters created an existing index
- Manual cleanup required when experimenting with parameters

**Example issue:**
```python
# Day 1: Create index with chunk_size=1000
processor = ONNXProcessor(persist_directory="./chroma_db_onnx")
processor.process_folder("pdfs/")  # Creates index with 1000-char chunks

# Day 2: Change chunk_size=500 in code
processor = ONNXProcessor(persist_directory="./chroma_db_onnx")
# Uses OLD index with 1000-char chunks! No warning!
```

**Current workaround:**
- Manually delete collection directories when changing parameters
- Or use different `persist_directory` per parameter set

## Data Flow

```
User Question
    ↓
[Router Agent]
    - Loads document_summaries.json
    - LLM selects top-k documents
    - Returns: ["doc1.pdf", "doc2.pdf", "doc3.pdf"]
    ↓
[Planner Agent]
    - Receives question + selected documents
    - LLM creates multi-step plan
    - Returns: RetrievalPlan with steps
    ↓
[Retriever Agent]
    - For each step in plan:
        1. Query ChromaDB (entire index)
        2. Filter by step.target_documents
        3. Fallback to partial match if needed
    - Combine results if needed
    - Returns: ExecutionResult with chunks
    ↓
[Orchestrator - Phase 5] (not yet implemented)
    - Receive chunks from all steps
    - Format context with sources
    - Call LLM to generate final answer
    - Return: Answer with citations
```

## Error Handling & Robustness

### Router Agent
- Validates document filenames against summaries
- Falls back to alphabetical selection if LLM response invalid
- Warns when documents not found
- Handles malformed JSON responses

### Planner Agent
- Supports both single-step and multi-step plans
- Defaults to simple plan if LLM output malformed
- Preserves question context throughout planning

### Retriever Agent
- Auto-indexes if processor empty
- Exact match → partial match fallback
- Verbose diagnostics for debugging
- Tracks filtered-out chunks
- Shows source of retrieved chunks

## Testing

**Test Script:** `test_retriever.py`

**Test Questions:**
- **EF_1:** "What are the rules for an ineligible risk?"
  - Tests distributed content retrieval (info across multiple sections)

- **EF_2:** "What is the premium for Tier 1, PC 5, $500k, 2% deductible?"
  - Tests multi-hop reasoning (requires combining info from multiple documents)

**Test Modes:**
- `--mode test`: Full pipeline (Router → Planner → Retriever)
- `--mode retriever`: Retriever only (manual plan input)
- `--mode interactive`: Interactive question input

## Configuration

All agents support configuration via dataclasses:

```python
@dataclass
class RouterConfig:
    model: str = "llama3.2"
    top_k_docs: int = 3
    temperature: float = 0.0
    max_tokens: int = 512

@dataclass
class PlannerConfig:
    model: str = "llama3.2"
    temperature: float = 0.0
    max_tokens: int = 2048

@dataclass
class RetrieverConfig:
    top_k_per_step: int = 5           # Chunks per retrieval step
    chunk_size: int = 1000             # Characters per chunk
    chunk_overlap: int = 200           # Overlap between chunks
    expand_context: int = 0            # ±N chunks for sliding window (0 = disabled)
    chunking_strategy: str = "document"  # "page" or "document"
    use_hybrid: bool = False           # Enable BM25 + semantic
    hybrid_alpha: float = 0.5          # Weight for semantic (adaptive at runtime)
    model: str = "llama3.2"
    temperature: float = 0.0
```

## Performance Characteristics

| Component | Latency | Memory | Notes |
|-----------|---------|--------|-------|
| ONNX Embeddings | ~50ms/chunk | ~500MB | CPU-optimized, 384 dims |
| BM25 Index Build | ~5-10s | ~100MB | In-memory, 1500 chunks |
| Router Agent | ~2-5s | ~200MB | Single LLM call |
| Planner Agent | ~3-8s | ~200MB | Complex reasoning, enumeration detection |
| Retriever Agent (semantic) | ~0.5-2s | ~500MB | Pre-filtered search |
| Retriever Agent (hybrid) | ~1-3s | ~600MB | BM25 + semantic + RRF merge |
| Sliding Window Expansion | ~0.5-1s | +200MB | ±2 chunks, deduplication |

**Bottlenecks:**
- LLM calls (Router, Planner) dominate latency (~70% of total time)
- Hybrid search adds ~2x retrieval time (still fast at <3s)
- Sliding window expansion adds context retrieval overhead
- Index size grows linearly with PDF count

## Key Design Decisions

1. **Pre-filtered search** (changed from search-then-filter)
   - **Current:** Filter documents BEFORE semantic search using ChromaDB where clause
   - **Rationale:** More efficient, better precision on tables, prevents cross-document interference
   - **Previous approach failed** on rate tables (returned prose instead of actual tables)

2. **Hybrid BM25 + Semantic Search**
   - **Chose:** Combine both with Reciprocal Rank Fusion
   - **Rationale:** Pure semantic insufficient for keyword-heavy queries, complementary strengths
   - **Adaptive weighting:** Query type determines alpha (enumeration vs reasoning)

3. **Sliding Window Expansion**
   - **Chose:** Retrieve with small chunks, expand with neighbors
   - **Rationale:** Avoids semantic dilution while providing complete context
   - **Alternative (large chunks) failed:** 3000-char chunks had worse retrieval despite containing content

4. **Configurable Chunking Strategy**
   - **Chose:** Support both page-level and document-level
   - **Rationale:** Document-level preserves tables, page-level provides semantic focus
   - **Use case dependent:** Different strategies optimal for different question types

5. **Enumeration-Aware Planning**
   - **Chose:** Detect "list all" patterns, use single-step broad retrieval
   - **Rationale:** Multi-step specific plans missed items from overly narrow queries
   - **Targets:** TOC, indexes, comprehensive lists for enumerations

6. **Separate agents vs Monolithic**
   - **Chose:** Separate Router, Planner, Retriever, Orchestrator agents
   - **Rationale:** Modularity, testability, clear responsibilities, easier debugging

7. **ONNX embeddings only** (simplified from dual-embedding)
   - **Chose:** Single embedding model (ONNX all-MiniLM-L6-v2)
   - **Rationale:** Performed best in experiments, simpler codebase (-286 lines), faster
   - **Removed:** Ollama embeddings after performance comparison

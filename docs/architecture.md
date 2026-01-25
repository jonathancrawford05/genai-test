# Multi-Agent RAG System Architecture

## Overview

This system implements a multi-agent Retrieval-Augmented Generation (RAG) pipeline for complex PDF question answering. It uses a three-agent architecture: Router → Planner → Retriever.

## System Components

### 1. Document Processing Layer

**Base Processor** (`src/base_processor.py`)
- Abstract base class for PDF processing
- Handles chunking with configurable size and overlap
- Manages ChromaDB collections for vector storage
- Default: 1000 char chunks, 200 char overlap

**Embedding Implementation:**
- **ONNXProcessor** (`src/onnx_processor.py`)
  - Model: all-MiniLM-L6-v2
  - Size: 79MB
  - Dimensions: 384
  - Runtime: CPU-optimized ONNX
  - Collection: `chroma_db/`
  - Chosen for performance after experimentation

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

**Purpose:** Create multi-step retrieval strategy for complex questions

**Input:**
- Question string
- Selected documents from Router

**Process:**
1. Analyzes question complexity
2. Determines if single-step or multi-step retrieval needed
3. Creates retrieval plan with:
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

**Key Features:**
- Handles both simple (1-step) and complex (multi-step) questions
- Assigns specific target documents per step
- Creates focused queries for each step
- Indicates if steps need to be combined for final answer

#### **Phase 4: Retriever Agent** (`src/agents/retriever_agent.py`)

**Purpose:** Execute retrieval plan and gather relevant chunks

**Input:**
- RetrievalPlan from Planner
- Initialized processor (ONNX or Ollama)

**Process:**
1. Executes each step in plan sequentially
2. For each step:
   - **Semantic search on ENTIRE index** via `processor.query()`
   - ChromaDB returns top_k chunks (default: 10) with metadata
   - **Post-search filtering by metadata:**
     - Filters chunks where `source_file in step.target_documents`
     - Exact filename matching first
   - **Fallback if zero matches:**
     - Attempts partial filename matching
     - Strips ID prefixes `(numbers-numbers)-` for comparison
     - Matches on core filename
3. Combines results if `plan.requires_combination=True`

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
- Auto-indexes PDFs if processor is empty
- Verbose mode shows retrieval diagnostics
- Two-tier filtering strategy:
  1. **Exact match:** `source_file == target_document`
  2. **Partial match:** Core filename substring matching (fallback)
- Tracks and reports filtered-out chunks
- Shows first 3 chunks per step in verbose mode

## Retrieval Architecture: Search-Then-Filter

### How It Works

**Question: Does the Retriever only search within selected documents, or search everything then filter?**

**Answer: Search entire index, then filter by metadata.**

```python
# Step 1: Semantic search on ENTIRE ChromaDB index
results = self.processor.query(
    query_text=step.query,
    top_k=10  # Default: retrieve 10 most similar chunks from all documents
)

# Step 2: Filter results by target documents (metadata filtering)
for doc_text, metadata in zip(results['documents'][0], results['metadatas'][0]):
    source_file = metadata.get('source_file', '')

    # Exact match first
    if source_file not in step.target_documents:
        filtered_out += 1
        continue

    chunks.append({'text': doc_text, 'source_file': source_file, ...})

# Step 3: Fallback to partial matching if exact filtering removed everything
if len(chunks) == 0 and filtered_out > 0:
    # Try partial matching: strip ID prefixes and match core filenames
    for doc_text, metadata in zip(results['documents'][0], results['metadatas'][0]):
        source_file = metadata.get('source_file', '')
        clean_source = source_file.split(')-', 1)[-1]
        clean_target = target_doc.split(')-', 1)[-1]

        if clean_target in clean_source or clean_source in clean_target:
            chunks.append({'text': doc_text, 'source_file': source_file, ...})
```

### Why This Approach?

**Pros:**
- Maximizes semantic recall - finds most relevant chunks regardless of document
- Simple implementation - ChromaDB's native query API
- Flexible filtering - can adjust filtering logic without changing search
- Enables serendipitous discovery - relevant chunks in "wrong" documents caught by fallback

**Cons:**
- May retrieve irrelevant chunks from non-target documents (then filter out)
- Slightly less efficient than pre-filtered search
- Filtering happens post-search, not during search

**Alternative (not implemented):**
```python
# Could use ChromaDB's where clause to pre-filter
results = self.collection.query(
    query_texts=[query_text],
    where={"source_file": {"$in": target_documents}},  # Pre-filter
    n_results=top_k
)
```
This would only search target documents, but requires exact filename matches and may miss relevant cross-document context.

## Index Management

### Current Approach

**Index Storage:**
- Each embedding type has separate persistent ChromaDB collection
- ONNX: `./chroma_db_onnx/`
- Ollama: `./chroma_db_ollama/`

**Index Lifecycle:**
```python
# 1. Check if collection exists
collection = client.get_collection(name="pdf_chunks_onnx")

# 2. Validate dimensions match expected (recreate if mismatch)
if stored_dims != expected_dims:
    client.delete_collection(name)
    collection = client.create_collection(name)

# 3. Check if collection has data
if collection.count() == 0:
    # Index PDFs from folder
    processor.process_folder(pdf_folder)
```

**Index Creation Parameters:**
- Chunk size: 1000 characters (hardcoded in base_processor.py:38)
- Chunk overlap: 200 characters (hardcoded in base_processor.py:39)
- Embedding model: Determined by processor class
- Metadata: source_file, page_number, chunk_index

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
    embedding_type: str = "onnx"  # or "ollama"
    top_k_per_step: int = 10
    chunk_size: int = 1000
    chunk_overlap: int = 200
```

## Performance Characteristics

| Component | Latency | Memory | Notes |
|-----------|---------|--------|-------|
| ONNX Embeddings | ~50ms/chunk | ~500MB | CPU-optimized, 384 dims |
| Router Agent | ~2-5s | ~200MB | Single LLM call |
| Planner Agent | ~3-8s | ~200MB | Complex reasoning |
| Retriever Agent | ~0.5-2s | ~500MB | ONNX embeddings |

**Bottlenecks:**
- LLM calls (Router, Planner) dominate latency
- Index size grows linearly with PDF count
- Retrieval speed depends on top_k parameters

## Key Design Decisions

1. **Search-then-filter vs Pre-filtered search**
   - Chose: Search entire index, filter by metadata
   - Rationale: Maximizes recall, simpler implementation, allows partial matching fallback

2. **Separate agents vs Monolithic**
   - Chose: Separate Router, Planner, Retriever agents
   - Rationale: Modularity, testability, clear responsibilities

3. **Exact + Partial matching**
   - Chose: Try exact match first, fallback to partial
   - Rationale: Handles LLM filename variations robustly

4. **Manual index management vs Automated**
   - Chose: Manual cleanup, no parameter tracking
   - Rationale: Simplicity for prototype, avoid premature complexity

5. **ONNX embeddings (standardized)**
   - Chose: Single embedding model (ONNX all-MiniLM-L6-v2)
   - Rationale: Performed best in experiments, simpler codebase, faster performance

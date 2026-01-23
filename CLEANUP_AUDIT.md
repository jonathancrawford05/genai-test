# Codebase Cleanup Audit Plan

## Executive Summary

After development, the codebase has accumulated:
- **Unused files**: sentence_transformer_processor.py (deprecated)
- **Significant duplication**: ~70% code overlap between processors
- **Redundant scripts**: Multiple experiment variants (experiment.py, enhanced_experiment.py, fixed_experiment.py)
- **Dead code**: Unused abstractions (rag_engine.py, vector_store.py)

## Current State Analysis

### Processor Files (6 files, 1,450 lines)

| File | Lines | Status | Issues |
|------|-------|--------|--------|
| `ultra_light_processor.py` | 322 | **KEEP** | Duplication with ollama processor |
| `ollama_embedding_processor.py` | 220 | **KEEP** | Duplication with ultra_light |
| `sentence_transformer_processor.py` | 215 | **REMOVE** | No longer used, replaced by ollama |
| `pdf_processor.py` | 173 | **EVALUATE** | Likely unused |
| `fast_pdf_processor.py` | 160 | **EVALUATE** | Likely unused |
| `vector_store.py` | 174 | **REMOVE** | Abstraction layer, unused |
| `rag_engine.py` | 186 | **REMOVE** | Abstraction layer, unused |

### Experiment Scripts (3 files, 1,076 lines)

| File | Lines | Status | Purpose |
|------|-------|--------|---------|
| `experiment.py` | 395 | **REMOVE** | Original (had bug) |
| `enhanced_experiment.py` | 344 | **EVALUATE** | Detailed CSV export |
| `fixed_experiment.py` | 337 | **KEEP** | Current working version |

### Query Scripts (5 files, 708 lines)

| File | Lines | Status | Purpose |
|------|-------|--------|---------|
| `query_rag.py` | 222 | **KEEP** | Unified interface (ollama/openai/anthropic) |
| `query_ollama.py` | 170 | **EVALUATE** | Ollama-specific, redundant? |
| `query_openai.py` | 148 | **EVALUATE** | OpenAI-specific, redundant? |
| `query_light.py` | 51 | **REMOVE** | Early prototype |
| `index_light.py` | 67 | **EVALUATE** | Still useful for quick indexing? |

### Other Files

| File | Lines | Status | Purpose |
|------|-------|--------|---------|
| `main.py` | 320 | **EVALUATE** | Uses old abstractions |
| `test_memory.py` | 62 | **REMOVE** | Development test |
| `test_questions.py` | 141 | **KEEP** | Useful for testing |
| `debug_pdf.py` | 54 | **REMOVE** | Debug script |

## Code Duplication Analysis

### Common Pattern Across Processors (~200 lines duplicated)

**Duplicated code:**
1. ✗ ChromaDB initialization (30 lines)
2. ✗ Folder processing logic (50 lines)
3. ✗ PDF reading with pypdf (40 lines)
4. ✗ Text chunking algorithm (80 lines)
5. ✗ Batch management (30 lines)
6. ✗ Query interface (20 lines)

**Only difference:**
- Embedding generation method (10-20 lines)

**Duplication percentage:** ~70% of code is identical

### Specific Duplication Examples

#### 1. Folder Processing (Identical in both)
```python
def process_folder(self, folder_path: str) -> int:
    pdf_folder = Path(folder_path)
    if not pdf_folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    pdf_files = sorted(pdf_folder.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDFs found in {folder_path}")

    print(f"\nFound {len(pdf_files)} PDFs to process")
    total_chunks = 0

    for file_num, pdf_path in enumerate(pdf_files, start=1):
        # ... (identical logic)
```

#### 2. Chunking Algorithm (Identical in both)
```python
def _chunk_text(self, text: str):
    start = 0
    prev_start = -1

    while start < len(text) and start != prev_start:
        end = min(start + self.chunk_size, len(text))

        # ... (identical logic with overlap, sentence boundaries)
```

#### 3. Query Interface (Identical in both)
```python
def query(self, query_text: str, top_k: int = 5):
    results = self.collection.query(
        query_texts=[query_text],
        n_results=top_k,
    )
    # ... (identical result processing)
```

## Proposed Refactoring

### Architecture: Base Class + Embedding Strategies

```
┌─────────────────────────────────────┐
│     BasePDFProcessor (Abstract)     │
│  - ChromaDB setup                   │
│  - Folder processing                │
│  - PDF reading                      │
│  - Text chunking                    │
│  - Batch management                 │
│  - Query interface                  │
│  - Abstract: _generate_embeddings() │
└─────────────────────────────────────┘
              ▲           ▲
              │           │
    ┌─────────┴───┐   ┌───┴──────────┐
    │ ONNXProcessor │   │OllamaProcessor│
    │ 20 lines      │   │ 30 lines      │
    └───────────────┘   └───────────────┘
```

### File Structure After Cleanup

```
src/
  __init__.py
  config.py
  base_processor.py          (NEW - 280 lines, shared logic)
  onnx_processor.py          (NEW - 20 lines, ONNX embeddings)
  ollama_processor.py        (NEW - 30 lines, Ollama embeddings)

Scripts:
  fixed_experiment.py        (KEEP - working version)
  query_rag.py               (KEEP - unified interface)
  index_light.py             (KEEP - quick indexing script)
  test_questions.py          (KEEP - testing)

Documentation:
  README.md
  USAGE.md
  RETRIEVAL_EXPLAINED.md
  etc.

Removed:
  ✗ src/ultra_light_processor.py
  ✗ src/ollama_embedding_processor.py
  ✗ src/sentence_transformer_processor.py
  ✗ src/pdf_processor.py
  ✗ src/fast_pdf_processor.py
  ✗ src/rag_engine.py
  ✗ src/vector_store.py
  ✗ experiment.py
  ✗ enhanced_experiment.py
  ✗ query_light.py
  ✗ query_ollama.py
  ✗ query_openai.py
  ✗ main.py
  ✗ test_memory.py
  ✗ debug_pdf.py
```

## Benefits

### Before Cleanup
- **7 processor files**, 1,450 lines
- **~70% code duplication**
- **3 experiment scripts**
- **5 query scripts**
- **Hard to maintain**: Changes need updates in multiple places
- **Confusing**: Which file to use?

### After Cleanup
- **3 processor files**, ~330 lines total
- **No duplication**: Shared logic in base class
- **1 experiment script**
- **1 query script**
- **Easy to maintain**: Changes in one place
- **Clear**: Base + two strategies

### Metrics
- **Lines of code**: 1,450 → 330 (77% reduction)
- **Files**: 7 → 3 (57% reduction)
- **Duplication**: 70% → 0%
- **Total Python files**: 20 → 7 (65% reduction)

## Implementation Plan

### Phase 1: Create Base Class (30 min)
1. Extract common code from ultra_light_processor.py
2. Create src/base_processor.py with abstract _generate_embeddings()
3. Test base class instantiation

### Phase 2: Refactor to Strategy Pattern (45 min)
4. Create src/onnx_processor.py (minimal, inherits base)
5. Create src/ollama_processor.py (minimal, inherits base)
6. Update fixed_experiment.py to use new processors
7. Test experiment runs with both embeddings

### Phase 3: Remove Dead Code (20 min)
8. Remove unused processor files
9. Remove buggy/redundant experiment scripts
10. Remove deprecated query scripts
11. Remove debug/test scripts

### Phase 4: Verify & Commit (15 min)
12. Run full experiment to verify functionality
13. Update imports in all remaining files
14. Update documentation
15. Git commit with cleanup summary

**Total estimated time: 2 hours**

## Risks & Mitigation

| Risk | Mitigation |
|------|------------|
| Breaking existing functionality | Keep test_questions.py, run tests after each phase |
| Lost useful code | Review each file before deletion |
| Import errors | Grep for all imports, update systematically |
| Performance regression | Benchmark before/after with same experiment |

## Success Criteria

- ✅ All tests pass (test_questions.py)
- ✅ Experiment runs successfully with both embeddings
- ✅ F1 scores match previous runs (no regression)
- ✅ No duplicate code between processors
- ✅ Code reduction >70%
- ✅ Clear, documented architecture

## Next Steps

1. Review this audit plan
2. Get user approval
3. Execute cleanup in phases
4. Commit changes
5. Update documentation

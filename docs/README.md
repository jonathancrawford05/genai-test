# Documentation

This folder contains architecture documentation and development notes for the multi-agent RAG system.

## Documents

### [architecture.md](./architecture.md)
**Comprehensive system architecture documentation**

Read this to understand:
- How the multi-agent pipeline works (Router → Planner → Retriever)
- Search-then-filter vs pre-filtered retrieval approach
- Index management and lifecycle
- Data flow through the system
- Error handling and robustness features
- Performance characteristics
- Key design decisions and tradeoffs

**Key sections:**
- **Retrieval Architecture: Search-Then-Filter** - Answers how filtering works (searches entire index, then filters by metadata, with partial matching fallback)
- **Index Management** - Current approach and limitations (no parameter tracking)
- **Data Flow** - Visual walkthrough of question → answer pipeline

### [future_refinements.md](./future_refinements.md)
**Tracked improvements and experimental ideas**

This is a living document tracking:
- Limitations discovered during development
- Proposed solutions with implementation estimates
- Experiments to run once features are implemented
- Open questions and design considerations

**Priority levels:**
- **High Priority** - Critical for experimentation and robustness (index tracking, configurable chunking)
- **Medium Priority** - Quality improvements (semantic chunking, citations, hybrid search)
- **Low Priority** - Nice-to-have optimizations (caching, async, table handling)

**Use this doc to:**
- Track technical debt
- Plan next development phases
- Document tradeoffs for future reference
- Guide experimentation work (Part 2 of test)

## Quick Reference

### How does filtering work?

From [architecture.md](./architecture.md#retrieval-architecture-search-then-filter):

1. **Semantic search on ENTIRE index** - ChromaDB returns top-k most similar chunks from all documents
2. **Filter by metadata** - Keep only chunks where `source_file in target_documents`
3. **Fallback to partial matching** - If exact match filters out everything, try matching core filenames (strip ID prefixes)

### Why not pre-filter before search?

Discussed in [architecture.md](./architecture.md#why-this-approach) and [future_refinements.md](./future_refinements.md#3-pre-filtered-search-option):

**Current approach (search-then-filter):**
- ✅ Maximizes semantic recall
- ✅ Simple implementation
- ✅ Allows partial matching fallback
- ❌ Retrieves some irrelevant chunks

**Alternative (pre-filtered search):**
- ✅ More efficient (only searches target docs)
- ❌ Requires exact filename matches
- ❌ May miss cross-document context

### Index parameter tracking issue?

See [future_refinements.md](./future_refinements.md#1-index-parameter-tracking--management):

**Problem:** Changing chunk_size or overlap in code doesn't recreate index - uses old index with wrong parameters

**Solution:** Track parameters in `index_config.json`, validate on load, auto-recreate if mismatch

## Navigation

```
docs/
├── README.md                    # This file - documentation index
├── architecture.md              # Current system architecture (read first!)
└── future_refinements.md        # Improvements and experiments (reference as needed)
```

## For Developers

**First time reading the codebase?**
1. Start with [architecture.md](./architecture.md) - understand the system
2. Run `test_retriever.py --mode test` to see it in action
3. Review [future_refinements.md](./future_refinements.md) before making changes

**Planning new features?**
1. Check [future_refinements.md](./future_refinements.md) - might already be documented
2. Add your idea to future_refinements.md with implementation estimate
3. Reference related improvements (e.g., "Depends on #1 index tracking")

**Running experiments?**
1. See [future_refinements.md > Experiments to Run](./future_refinements.md#experiments-to-run)
2. Use index parameter tracking (#1) to avoid silent bugs
3. Implement evaluation framework (#12) for systematic comparison

## Questions?

If documentation doesn't answer your question, add it to [future_refinements.md > Open Questions](./future_refinements.md#open-questions).

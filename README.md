# Generative AI Test

## Role Context

The Generative AI role at ZestyAI focuses on building, evaluating, and improving **Retrieval-Augmented Generation (RAG)** pipelines and agentic workflows that work with both structured and unstructured data. It involves implementing solutions, running experiments, developing evaluation frameworks, and iterating to improve performance.

You are **not expected to know everything**. There will be opportunities to learn new tools, techniques, and frameworks as you go. You will also encounter challenges that don’t have predefined solutions, and part of the role is figuring out how to approach these problems.

The purpose of this test is to give you a taste of the types of problems you might see at ZestyAI. The questions here were carefully designed to reflect **real work** in this role. If you enjoy this test, there’s a good chance you’ll enjoy the work itself.

## Overview

The goal of this test is to create a prototype solution that can answer questions about PDF documents. The PDFs may be long and contain complex content such as tables, lists, and text spread across multiple pages.

This test has **two parts**:

* **Part 1:** Build a function to answer questions about PDFs.
* **Part 2:** Build a basic experimentation harness to test multiple solution ideas from Part 1.

This test focuses on prototyping a solution. You are not expected to deliver production-level code. Your solution is meant to be a prototype that can run in a notebook or a Python script.

---

## Part 1: PDF Question Answering

### Objective

Create a function that answers questions based on PDF documents. The function will receive a folder containing the documents and a specific question, and it should answer that question correctly. You may use any agentic design pattern of your choice.

You will be given 2 questions and their expected answers.

### Updated Function Signature

```python
def answer_pdf_question(question: str, pdfs_folder: str) -> str:
    """
    Inputs:
        question: A question about the content of the PDFs.
        pdfs_folder: Path to a folder containing all the PDFs needed to answer the question.
    Output:
        answer: Answer to the question.
    """
    pass
```

Here’s an example of how you would call the function:

```python
# Example 1
question_1 = "List all rating plan rules"
pdfs_folder_1 = "artifacts/1"  
answer_1 = answer_pdf_question(question_1, pdfs_folder_1)
```

### Requirements & Guidance

1. **Focus on accuracy:** Ensure your function produces correct answers for the provided PDFs and questions, while also being designed to generalize beyond these specific cases, not relying on one-off solutions.

2. You may use **any tools, frameworks, or LLM calls**. Multiple steps, agents, or tool integrations are acceptable.

3. It is acceptable to use AI tools (e.g., ChatGPT, Cursor, etc.) during development. If you do, please keep a record of the prompts and steps you used. This helps us understand your reasoning and how you arrived at your solution.

4. Optimizing for context efficiency (token usage) and latency (speed) is encouraged, but accuracy remains the top priority. You should also consider the **scalability** and **cost** of your solution (e.g., handling large numbers of PDFs).

You may create API keys as needed, but please let us know which ones you used so we can reproduce your results.

> **Note:** If you are not able to achieve perfect accuracy for all questions, that is fine as long as your approach is systematic and the evaluation metrics clearly show where the solution performs well and where it struggles.

---

## Implementation Overview

This section describes the final architecture and key features of the implemented solution.

### Multi-Agent RAG Architecture

The solution uses a **4-agent pipeline** to decompose complex question-answering into specialized steps:

1. **Router Agent** - Selects relevant documents from the 22 PDFs using pre-generated summaries
2. **Planner Agent** - Creates retrieval strategies, with special handling for enumeration questions
3. **Retriever Agent** - Executes retrieval with hybrid BM25+semantic search and sliding window expansion
4. **Orchestrator Agent** - Synthesizes final answers from retrieved context

### Key Features

#### 1. Hybrid Search (BM25 + Semantic)

Combines keyword-based BM25 with semantic embeddings for robust retrieval:
- **Semantic search** - Good for conceptual queries and reasoning
- **BM25 search** - Excellent for exact term matching, lists, enumerations
- **Adaptive weighting** - Automatically adjusts based on query type:
  - Enumeration queries ("list all"): 70% BM25, 30% semantic
  - Reasoning queries: 30% BM25, 70% semantic
- **Reciprocal Rank Fusion (RRF)** - Merges both rankings optimally

#### 2. Full-Document Chunking

Solves multi-page table splitting issues:
- Extracts entire PDF text first, then chunks (chunks can span pages)
- Preserves tables that cross page boundaries (critical for TOC tables)
- Tracks page ranges (e.g., "3-4") for multi-page chunks
- Alternative: Page-level chunking available for better semantic focus

#### 3. Sliding Window Context Expansion

Retrieves with precision, expands for context:
- Retrieves using small chunks (1000 chars) for precise similarity matching
- Expands with ±N neighboring chunks for complete context
- Prevents semantic dilution from large chunks
- Configurable expansion window (default: ±2 chunks)

#### 4. Enumeration-Aware Planning

Special handling for list/enumeration questions:
- Detects patterns: "list all", "enumerate", "what are all"
- Uses single-step broad retrieval instead of multi-step specific queries
- Targets table of contents, indexes, comprehensive lists
- Prevents missing items from overly specific sub-queries

#### 5. Pre-Filtered Search

Filters documents BEFORE semantic search, not after:
- More efficient - only searches target documents
- Better precision - finds chunks in tables and structured data
- Prevents cross-document interference

### Technology Stack

- **Embeddings**: ONNX (all-MiniLM-L6-v2) - 384 dimensions, CPU-optimized
- **Vector DB**: ChromaDB with persistent storage
- **LLM**: Ollama (llama3.2) for local inference
- **BM25**: rank-bm25 library for keyword search
- **PDF Processing**: pypdf for text extraction

### Performance Characteristics

- **Indexing**: ~30-60 seconds for 22 PDFs (1500+ chunks)
- **Query Time**: 30-90 seconds per question (4 agent steps)
- **Accuracy**: High precision on both enumeration and reasoning questions
- **Cost**: $0 (fully local with Ollama)

---

## Part 2: Experimentation Harness

### Objective

Design a small framework to evaluate multiple iterations or approaches from Part 1. Experimentation is a core part of improving generative AI systems.

### Implementation

The experimentation harness tests **6 variations** across multiple dimensions:

#### Variations Tested

1. **Baseline** - Standard configuration (document-level chunking, no expansion)
2. **High Depth** - Increased retrieval (top_k=10 per step)
3. **Conservative** - Reduced retrieval for speed (top_k=3 per step)
4. **Sliding Window** - Document-level + ±2 chunk expansion
5. **Page Window** - Page-level chunking + ±2 chunk expansion
6. **Hybrid Search** - Page-level + expansion + BM25+semantic hybrid search

#### Evaluation Metrics

1. **Exact Match** - Binary: Does answer exactly match expected output?
2. **Partial Match Score** - Keyword overlap for list questions (e.g., 35/35 rules found)
3. **Execution Time** - Latency in seconds
4. **Token Efficiency** - Context size used

#### Features

- Runs all variations on all test questions
- Generates CSV and Markdown comparison reports
- Tracks successes and failures for iteration
- Isolated configurations prevent cross-contamination


> When recording results for each variation, be sure to capture **both successes and failures**. The metrics should clearly indicate where the solution fails so we can assess limitations and potential improvements.

---

## Example Questions

You will find the PDFs and the CSV containing the questions inside the `artifacts` folder.

**Note:** For details about how expected answers were determined, please refer to the README file in the artifacts folder.

### CSV Columns

The questions CSV contains:

* **id** — Unique identifier for each question
* **question** — The question to answer
* **expected_output** — The expected answer
* **PDF Folder** — The folder containing the relevant PDF(s)

---

## Deliverables

1. A Python file or notebook implementing **Part 1** and **Part 2**.
2. Example runs showing answers for all provided PDFs and questions.
3. (Optional) Notes on prompts and experimentation steps if you used AI tools.
4. **Documentation of decisions:** Include comments describing key decisions (e.g., evaluation metric choice, PDF processing approach, agent design). This helps us understand your reasoning.
5. Ensure your code is fully runnable by including clear setup instructions.

---

## Thanks

We recognize that this test requires a significant amount of work. We hope you find it interesting and challenging. Thank you for your time and effort.

This test is a work in progress. If anything is unclear or you have any suggestions, please reach out.


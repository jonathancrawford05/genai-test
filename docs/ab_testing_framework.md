# A/B Testing Framework for EF1 & EF2 (Ollama + llama3.2)

## Goal
Create a repeatable A/B testing framework for the EF1 (enumeration) and EF2 (calculation) questions that evaluates how decoding parameters affect output quality in both **creative** and **high-precision** scenarios, using the existing agentic system. This framework is designed around **Ollama** running **llama3.2**, so it only relies on parameters that Ollama exposes today.

## Scope & Assumptions
- **Model**: `llama3.2` via Ollama.
- **Available decoding parameters** (Ollama options): `temperature`, `top_k`, `top_p`, `repeat_penalty`, `repeat_last_n`, `num_predict`, `num_ctx`, `seed`, `stop`.
- **Beam search**: Not exposed by Ollama for llama3.2. We'll emulate "beam-like" behavior with **deterministic decoding** plus **multi-sample voting** (see Beam Search Emulation).
- **Tasks**:
  - EF1: Enumeration/listing.
  - EF2: Calculation.
- **Two scenarios**:
  - **Creative**: prioritize diversity and expressive language.
  - **High-precision**: prioritize factual accuracy and consistency.

## A/B Testing Design
### Variants
Each EF question is evaluated across **two A/B tracks**:

1. **Creative Track (A/B)**
   - **A (baseline)**: moderate creativity, controlled randomness.
   - **B (variant)**: higher creativity.

2. **High-Precision Track (A/B)**
   - **A (baseline)**: deterministic, strict decoding.
   - **B (variant)**: slightly relaxed decoding to reduce brittleness.

### Fixed Variables (hold constant)
- Agentic pipeline (router → planner → retriever → orchestrator).
- Retrieval settings (chunking, `top_k_docs`, etc.).
- Prompt and system instructions for EF1/EF2.
- Document set.

### Independent Variables (test)
- **temperature**
- **top_k**
- **top_p**
- **repeat_penalty**
- **seed** (for reproducibility)
- **num_predict** (cap output length)

## Parameter Grid (Ollama + llama3.2)
Use a **small factorial grid** to keep tests efficient.

### Creative Track
| Variant | temperature | top_k | top_p | repeat_penalty | num_predict | seed |
|---------|-------------|-------|-------|----------------|-------------|------|
| A (baseline) | 0.7 | 40 | 0.9 | 1.1 | 512 | 42 |
| B (variant)  | 0.9 | 80 | 0.95 | 1.0 | 768 | 43 |

### High-Precision Track
| Variant | temperature | top_k | top_p | repeat_penalty | num_predict | seed |
|---------|-------------|-------|-------|----------------|-------------|------|
| A (baseline) | 0.0 | 1 | 1.0 | 1.2 | 512 | 123 |
| B (variant)  | 0.2 | 10 | 0.9 | 1.2 | 512 | 124 |

## Beam Search Emulation (Ollama-Compatible)
Since Ollama doesn't support beam search for llama3.2, emulate it via **multi-sample voting**:
1. Run the same prompt **N times** with the same deterministic settings (e.g., `temperature=0`, `top_k=1`).
2. Use **self-consistency**: choose the most common answer or highest-scoring answer by your evaluator.
3. Track **variance** across runs as an uncertainty signal.

Recommended: **N = 3–5** for EF2 (calculation) and **N = 5–7** for EF1 (enumeration).

## Evaluation Metrics
### EF1 (Enumeration)
- **Coverage**: % of expected items included.
- **Precision**: % of items that are correct (avoid hallucinated items).
- **Order/format compliance**: adherence to a bullet list with `*` prefix.
- **Traceability**: % of items with correct source/page references (if required).

### EF2 (Calculation)
- **Answer correctness**: exact match (expected $604).
- **Intermediate correctness**: presence of correct base rate, factor, deductible, formula.
- **Citation accuracy**: references to correct exhibits/pages.

### Creative Track (cross-cutting)
- **Fluency**: readability and naturalness (human or heuristic score).
- **Diversity**: lexical or semantic diversity across runs (e.g., distinct n-grams or embedding variance).

### High-Precision Track (cross-cutting)
- **Determinism**: identical outputs across 3 runs (with same seed).
- **Hallucination rate**: any facts not present in context.

## Experimental Workflow (Agentic System)
1. **Run EF1 & EF2 through baseline decoding** (Creative A, Precision A).
2. **Run EF1 & EF2 through variant decoding** (Creative B, Precision B).
3. **Collect outputs + metadata** (parameters, run ID, seed, timing).
4. **Score** with existing evaluation scripts (e.g., EF2 component scorer) and add EF1 list scorer if needed.
5. **Compare A vs B** within each track and question.

### Recommended Output Schema
```json
{
  "run_id": "ef1-creative-a-001",
  "question_id": "EF1",
  "track": "creative",
  "variant": "A",
  "model": "llama3.2",
  "ollama_options": {
    "temperature": 0.7,
    "top_k": 40,
    "top_p": 0.9,
    "repeat_penalty": 1.1,
    "num_predict": 512,
    "seed": 42
  },
  "answer": "...",
  "metrics": {
    "coverage": 0.82,
    "precision": 0.90
  },
  "timestamp": "2024-..."
}
```

## Mapping to the Prompt (Creative vs High-Precision)
Use the lab prompt as the **global system instruction**, then vary decoding per track.

### Creative Scenario
- Emphasize **creative blogs** in the system prompt.
- Use Creative Track settings.
- Score for expressiveness + acceptable factuality.

### High-Precision Scenario
- Emphasize **technical summaries** and **customer support responses**.
- Use High-Precision Track settings.
- Score for correctness, determinism, and low hallucination rate.

## Decision Rules
- **Creative Track winner**: higher diversity + acceptable factuality (coverage ≥ 80%, hallucination rate ≤ 5%).
- **High-Precision winner**: exact EF2 match + ≥ 90% EF1 coverage with zero hallucinations.

## Reporting Template
Summarize A/B results per track/question:
- Best parameters
- Metric deltas (A vs B)
- Sample outputs (1–2 per track)
- Any failure modes

## Experiment Results (2026-01-28)

### Summary (40 tests: 2 questions × 2 tracks × 5 configs × 2 variants)

| Question | Track | Variant A | Variant B | Delta | Winner |
|----------|-------|-----------|-----------|-------|--------|
| EF_1 | Creative | **32.0%** | 17.7% | -14.3 | **A** |
| EF_1 | Precision | 17.2% | 17.7% | +0.6 | B |
| EF_2 | Creative | 53/100 | **57/100** | +4.0 | **B** |
| EF_2 | Precision | 53/100 | **55/100** | +2.0 | **B** |

### Best Individual Results
- **EF_1**: 82.9% coverage (creative-A, chunk=2000, top_k=5)
- **EF_2**: 60/100 (multiple configs with creative track)

### Key Findings

1. **EF_1 (Enumeration)**:
   - Moderate temperature (0.7) significantly outperforms high creativity (0.9)
   - Pure deterministic decoding (temp=0.0) underperforms moderate creativity
   - Best results achieved with creative-A settings

2. **EF_2 (Calculation)**:
   - Slightly higher temperature (0.9) improves scores
   - Variant B consistently wins across both tracks
   - Deterministic decoding (temp=0.0) is not optimal for calculations

3. **General Insights**:
   - Optimal decoding parameters are task-dependent
   - One-size-fits-all decoding settings are suboptimal
   - Enumeration tasks prefer controlled randomness, calculations benefit from exploration

### Results File
Full results available in: `results/ab_test_20260128_210315.json`

---

## Future Enhancements (Not Implemented)
- **Beam search emulation** via multi-sample voting
- **LLM-as-judge** for qualitative evaluation
- **Additional metrics**: fluency, diversity, hallucination rate
- **Cross-encoder re-ranking** for improved retrieval

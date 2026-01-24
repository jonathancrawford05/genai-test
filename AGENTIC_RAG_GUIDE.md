# Agentic RAG System - Implementation Guide

## Overview

Multi-agent RAG architecture for complex question answering that requires:
- Document-level routing (which documents are relevant?)
- Multi-step planning (how to retrieve information?)
- Strategic retrieval (execute the plan)

## Architecture

```
Question
   ↓
┌──────────────┐
│ Router Agent │  ← Uses document summaries
└──────────────┘
   ↓ [Selected Documents]
┌───────────────┐
│ Planner Agent │  ← Formulates strategy
└───────────────┘
   ↓ [Retrieval Plan]
┌─────────────────┐
│ Retriever Agent │  ← Executes retrieval
└─────────────────┘
   ↓
Answer
```

### Agent Roles

1. **Router Agent**:
   - Input: Question + document summaries
   - Output: Subset of relevant documents (e.g., 3 of 22)
   - Purpose: Reduce search space, focus on relevant documents

2. **Planner Agent**:
   - Input: Question + selected documents
   - Output: Multi-step retrieval strategy
   - Purpose: Break complex questions into steps
   - Conversations: Can refine strategy through dialogue

3. **Retriever Agent**:
   - Input: Retrieval plan + document chunks
   - Output: Extracted information
   - Purpose: Execute retrieval steps
   - Conversations: Can iterate with planner if needed

## Phase 1: Document Summaries (CURRENT)

### Status: ✅ Ready to Use

Generate structured summaries for document routing.

### Usage

```bash
# Generate summaries for all PDFs
python generate_summaries.py

# Custom options
python generate_summaries.py \
  --folder artifacts/1 \
  --output artifacts/document_summaries.json \
  --model llama3.2 \
  --sample-pages 3
```

### Output Structure

```json
{
  "document_name.pdf": {
    "filename": "document_name.pdf",
    "total_pages": 108,
    "generated_by": "llama3.2",
    "summary": "1-2 sentence overview",
    "use_for": "What questions this document can answer",
    "key_topics": ["topic1", "topic2", "topic3"],
    "document_type": "rules_manual | rate_pages | checklist | actuarial_memo | forms"
  }
}
```

### Manual Review

After generation:

1. **Review**: Open `artifacts/document_summaries.json`
2. **Edit**: Refine summaries for accuracy
3. **Add notes**: Use for important context
4. **Categorize**: Ensure `document_type` is correct

**Example edits:**

```json
{
  "CT_Rules_Manual.pdf": {
    "summary": "Connecticut homeowners underwriting rules effective 8/18/25",
    "use_for": "Questions about CT underwriting, eligibility, property restrictions, or endorsement rules",
    "key_topics": ["underwriting", "eligibility", "property types", "endorsements", "occupancy"],
    "document_type": "rules_manual",
    "review_notes": "Primary source for all underwriting decisions"
  }
}
```

## Phase 2: Router Agent (COMPLETE ✅)

### Status: Ready to Use

Document routing agent that selects top-k most relevant documents from summaries.

### Implementation

```python
from src.agents.router_agent import RouterAgent, RouterConfig

# Initialize router
router = RouterAgent(
    summaries_path="artifacts/document_summaries.json",
    config=RouterConfig(
        model="llama3.2",
        top_k_docs=3,
        temperature=0.0
    )
)

# Select documents for a question
selected_docs = router.select_documents(
    question="What are the rules for ineligible risks?",
    top_k=3,
    verbose=True
)
# Returns: ["CT_Rules_Manual.pdf", "Underwriting_Checklist.pdf", ...]
```

### Configuration

```python
@dataclass
class RouterConfig:
    model: str = "llama3.2"           # LLM for routing
    top_k_docs: int = 3                # How many docs to select
    temperature: float = 0.0           # Deterministic
    max_tokens: int = 512              # Response length
```

### Testing

```bash
# Run test suite with sample questions
python test_router.py --mode test

# Interactive mode - test with custom questions
python test_router.py --mode interactive
```

### How It Works

1. **Load summaries**: Reads `artifacts/document_summaries.json`
2. **Format prompt**: Presents all document summaries to LLM
3. **LLM selection**: llama3.2 analyzes and selects most relevant documents
4. **Parse response**: Extracts JSON array of filenames
5. **Validate**: Ensures selected documents exist in summaries

### Features

- ✅ Strategic document selection (not just keyword matching)
- ✅ Considers document types (rules vs rates vs forms)
- ✅ Fallback handling if LLM fails
- ✅ Conversation history support (via BaseAgent)
- ✅ Verbose mode for debugging
- ✅ Configurable top-k selection

## Phase 3: Planner Agent

### Planned Implementation

```python
class PlannerAgent:
    def create_plan(self, question: str, documents: List[str]) -> Plan:
        """
        Formulate multi-step retrieval strategy.

        Returns: Structured plan with steps
        """
```

## Phase 4: Retriever Agent

### Planned Implementation

```python
class RetrieverAgent:
    def execute_plan(self, plan: Plan, processor) -> Dict:
        """
        Execute retrieval plan using existing processors.

        Returns: Retrieved information
        """
```

## Integration with Experiments

### Baseline Comparison

```python
# Baseline: Simple RAG (no agents)
config = AgentConfig(
    enable_router=False,
    enable_planner=False,
)

# Router only
config = AgentConfig(
    enable_router=True,
    enable_planner=False,
)

# Full pipeline
config = AgentConfig(
    enable_router=True,
    enable_planner=True,
)
```

### Experiment Harness Integration

```bash
# Compare configurations
python fixed_experiment.py \
  --agent-config baseline \
  --embeddings onnx \
  --llms llama3.2

python fixed_experiment.py \
  --agent-config full \
  --embeddings onnx \
  --llms llama3.2
```

## Example Questions

### EF_1: Distributed Information

**Question**: "What are the rules for an ineligible risk?"

**Challenge**: Answer distributed across section headings throughout document

**Agent Strategy**:
1. **Router**: Select rules manual
2. **Planner**: "Scan for section headings containing 'ineligible' or 'restrictions'"
3. **Retriever**: Extract all relevant headings, combine into answer

### EF_2: Multi-hop Reasoning

**Question**: "Calculate premium for Tier 1, Protection 5, $500k coverage, 2% deductible"

**Challenge**: Requires lookup from multiple documents

**Agent Strategy**:
1. **Router**: Select rate pages + deductible factor table
2. **Planner**:
   - Step 1: Find base rate for Tier 1, Protection 5, $500k
   - Step 2: Find deductible multiplier for 2%
   - Step 3: Calculate: base_rate × deductible_factor
3. **Retriever**: Execute each step, combine results

## Development Status

- [x] Phase 1: Document Summary Generation
- [x] Phase 2: Router Agent
- [ ] Phase 3: Planner Agent
- [ ] Phase 4: Retriever Agent
- [ ] Phase 5: Orchestrator
- [ ] Phase 6: Experiment Integration
- [ ] Phase 7: Evaluation & Iteration

## Next Steps

1. ~~**Run summary generation** on your machine~~ ✅ DONE
   ```bash
   python generate_summaries.py
   ```

2. ~~**Review and refine** summaries~~ ✅ DONE
   - Summaries in `artifacts/document_summaries.json`

3. **Test Router Agent**:
   ```bash
   # Run test suite
   python test_router.py --mode test

   # Interactive testing
   python test_router.py --mode interactive
   ```

4. **Proceed to Phase 3**: Implement Planner Agent

5. **Test incrementally**: Compare baseline vs router-only vs full pipeline

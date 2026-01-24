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

## Phase 3: Planner Agent (COMPLETE ✅)

### Status: Ready to Use

Strategy formulation agent that creates multi-step retrieval plans.

### Implementation

```python
from src.agents.planner_agent import PlannerAgent, PlannerConfig, RetrievalPlan

# Initialize planner
planner = PlannerAgent(
    config=PlannerConfig(
        model="llama3.2",
        temperature=0.0,
        enable_conversation=True
    )
)

# Create plan (after Router selects documents)
plan = planner.create_plan(
    question="Calculate premium for Tier 1, PC 5, $500k, 2% deductible",
    selected_documents=["rate_pages.pdf", "factors.pdf"],
    summaries=router.summaries,
    verbose=True
)

# Plan structure:
# plan.strategy - High-level approach
# plan.steps - List of RetrievalStep objects
# plan.requires_combination - True if steps must be combined
```

### Plan Structure

```python
@dataclass
class RetrievalPlan:
    question: str
    strategy: str  # High-level approach
    steps: List[RetrievalStep]
    success_criteria: str
    requires_combination: bool  # Multi-hop reasoning?

@dataclass
class RetrievalStep:
    step_number: int
    description: str
    target_documents: List[str]
    query: str
    expected_output: str
```

### Configuration

```python
@dataclass
class PlannerConfig:
    model: str = "llama3.2"
    temperature: float = 0.0
    max_tokens: int = 1024
    enable_conversation: bool = True
```

### Testing

```bash
# Run test suite (Router → Planner integration)
python test_planner.py --mode test

# Interactive mode
python test_planner.py --mode interactive

# Test plan refinement (conversational)
python test_planner.py --mode refinement
```

### How It Works

1. **Receive input**: Question + selected documents from Router
2. **Analyze question**: Break down into logical retrieval steps
3. **Create plan**: Structured steps with queries and target docs
4. **Determine complexity**: Single-step vs multi-hop reasoning
5. **Return plan**: Ready for Retriever agent to execute

### Example Plans

**Simple Question (EF_1):**
```
Question: "What are the rules for ineligible risks?"
Strategy: "Direct lookup of ineligibility rules from rules manual"
Steps:
  1. Search rules manual for "ineligible risk" sections
     Query: "ineligible risk rules restrictions"
     Expected: List of ineligibility criteria
Requires combination: False
```

**Complex Question (EF_2):**
```
Question: "Premium for Tier 1, PC 5, $500k, 2% deductible"
Strategy: "Multi-step lookup and calculation"
Steps:
  1. Find base rate for Tier 1, PC 5, $500k
     Query: "Tier 1 Protection Class 5 $500,000 base rate"
     Docs: rate_pages.pdf
  2. Find deductible factor for 2%
     Query: "2% deductible factor multiplier"
     Docs: rate_pages.pdf
  3. Calculate final premium
     Expected: base_rate × deductible_factor
Requires combination: True
```

### Conversational Refinement

```python
# Refine plan based on feedback
refined_plan = planner.refine_plan(
    plan=original_plan,
    feedback="Step 2 should search factors.pdf not rates.pdf",
    verbose=True
)
```

### Features

- ✅ Multi-step plan creation
- ✅ Single-step vs multi-hop detection
- ✅ Document targeting per step
- ✅ Specific query formulation
- ✅ Conversational refinement
- ✅ JSON serialization (to_dict/from_dict)
- ✅ Fallback to simple plan if parsing fails
- ✅ Integration with Router

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
- [x] Phase 3: Planner Agent
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

3. ~~**Test Router Agent**~~ ✅ DONE
   ```bash
   python test_router.py --mode test
   ```

4. **Test Router → Planner pipeline**:
   ```bash
   # Run integrated test suite
   python test_planner.py --mode test

   # Interactive testing
   python test_planner.py --mode interactive
   ```

5. **Proceed to Phase 4**: Implement Retriever Agent

6. **Test incrementally**: Compare baseline vs router-only vs full pipeline

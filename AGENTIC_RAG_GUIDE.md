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

## Phase 4: Retriever Agent (COMPLETE ✅)

### Status: Ready to Use

Execution agent that performs actual retrieval using existing processors.

### Implementation

```python
from src.agents.retriever_agent import RetrieverAgent, RetrieverConfig
from src.onnx_processor import ONNXProcessor

# Initialize processor (reuse existing index)
processor = ONNXProcessor(
    persist_directory="./chroma_db_onnx",
    collection_name="pdf_documents"
)

# Initialize retriever
retriever = RetrieverAgent(
    processor=processor,
    config=RetrieverConfig(
        embedding_type="onnx",
        top_k_per_step=5,
        model="llama3.2"
    )
)

# Execute plan (from Planner)
execution_result = retriever.execute_plan(plan, verbose=True)

# Get answer context
context = retriever.get_answer_context(execution_result, max_chunks=5)
```

### Result Structures

```python
@dataclass
class RetrievalResult:
    step_number: int
    query: str
    chunks: List[Dict]  # Retrieved chunks with metadata
    num_chunks: int
    target_documents: List[str]

@dataclass
class ExecutionResult:
    question: str
    plan_strategy: str
    step_results: List[RetrievalResult]
    requires_combination: bool
    combined_context: Optional[str]  # If multi-hop
```

### Configuration

```python
@dataclass
class RetrieverConfig:
    embedding_type: str = "onnx"  # "onnx" or "ollama"
    top_k_per_step: int = 5
    model: str = "llama3.2"  # For combination/synthesis
    temperature: float = 0.0
```

### Testing

```bash
# Run full pipeline test (Router → Planner → Retriever)
python test_retriever.py --mode test

# Interactive mode
python test_retriever.py --mode interactive

# Test retriever only (with manual plan)
python test_retriever.py --mode retriever-only
```

### How It Works

1. **Receive plan**: Gets RetrievalPlan from Planner
2. **Execute steps**: Sequentially executes each step
   - Query processor with step query
   - Filter by target documents
   - Collect top-k chunks per step
3. **Combine if needed**: If `requires_combination=True`
   - Aggregates contexts from all steps
   - Optionally synthesizes combined context
4. **Return results**: ExecutionResult with all retrieved information

### Example Execution

**Simple Question (EF_1):**
```
Plan: 1 step, no combination
Step 1: Query "ineligible risk rules restrictions"
        Target: rules_manual.pdf
        Retrieved: 5 chunks

Result:
  - 5 chunks from rules manual
  - No combination needed
  - Context ready for answer generation
```

**Complex Question (EF_2):**
```
Plan: 3 steps, requires combination
Step 1: Query "Tier 1 Protection Class 5 $500,000 base rate"
        Target: rate_pages.pdf
        Retrieved: 5 chunks

Step 2: Query "2% deductible factor multiplier"
        Target: rate_pages.pdf
        Retrieved: 5 chunks

Step 3: Combine information
        Expected: base_rate × deductible_factor

Result:
  - 10 chunks total (5 per step)
  - Combined context created
  - Ready for calculation/answer generation
```

### Features

- ✅ Step-by-step plan execution
- ✅ Integration with existing processors (ONNX/Ollama)
- ✅ Document filtering per step
- ✅ Top-k retrieval per step
- ✅ Multi-hop combination
- ✅ Chunk metadata preservation
- ✅ Answer context generation
- ✅ Verbose execution logging

### Full Pipeline Example

```python
# Complete Router → Planner → Retriever pipeline
from src.agents import RouterAgent, PlannerAgent, RetrieverAgent
from src.onnx_processor import ONNXProcessor

# Initialize
router = RouterAgent(summaries_path="artifacts/document_summaries.json")
planner = PlannerAgent()
processor = ONNXProcessor(persist_directory="./chroma_db_onnx")
retriever = RetrieverAgent(processor=processor)

# Execute pipeline
question = "What are the rules for ineligible risks?"

# Step 1: Route
selected_docs = router.select_documents(question, top_k=3)

# Step 2: Plan
plan = planner.create_plan(question, selected_docs, router.summaries)

# Step 3: Retrieve
result = retriever.execute_plan(plan, verbose=True)

# Step 4: Get answer context
context = retriever.get_answer_context(result)

# Now context can be used with LLM for final answer generation
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
- [x] Phase 4: Retriever Agent
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

4. ~~**Test Router → Planner pipeline**~~ ✅ DONE
   ```bash
   python test_planner.py --mode test
   ```

5. **Test Full Pipeline (Router → Planner → Retriever)**:
   ```bash
   # Run complete pipeline test
   python test_retriever.py --mode test

   # Interactive testing
   python test_retriever.py --mode interactive
   ```

6. **Proceed to Phase 5**: Implement Orchestrator (wires agents together)

7. **Phase 6**: Integrate with experiment harness for A/B testing

8. **Test incrementally**: Compare baseline vs router-only vs full pipeline

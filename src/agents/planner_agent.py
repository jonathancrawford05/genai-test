"""
Planner Agent for multi-step retrieval strategy formulation.

Creates structured plans for complex question answering.
"""
import json
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict, field

from .base_agent import BaseAgent


@dataclass
class RetrievalStep:
    """Single step in a retrieval plan."""
    step_number: int
    description: str
    target_documents: List[str]
    query: str
    expected_output: str


@dataclass
class RetrievalPlan:
    """
    Structured plan for multi-step retrieval.

    Contains:
    - Question being answered
    - Strategy overview
    - Ordered list of retrieval steps
    - Success criteria
    """
    question: str
    strategy: str  # High-level approach
    steps: List[RetrievalStep]
    success_criteria: str
    requires_combination: bool = False  # True if steps need to be combined


@dataclass
class PlannerConfig:
    """Configuration for Planner Agent."""
    model: str = "llama3.2"
    temperature: float = 0.0
    max_tokens: int = 1024
    enable_conversation: bool = True  # Allow refinement dialogue


class PlannerAgent(BaseAgent):
    """
    Planning agent that formulates multi-step retrieval strategies.

    Receives:
    - Question
    - Selected documents from Router
    - Document summaries (for context)

    Produces:
    - Structured RetrievalPlan with ordered steps
    - Each step specifies what to retrieve and from which docs

    Example:
        planner = PlannerAgent()
        plan = planner.create_plan(
            question="Calculate premium...",
            selected_documents=["rates.pdf", "factors.pdf"],
            summaries=summaries_dict
        )
    """

    def __init__(self, config: Optional[PlannerConfig] = None):
        """
        Initialize Planner Agent.

        Args:
            config: Planner configuration
        """
        self.config = config or PlannerConfig()
        super().__init__(model=self.config.model, temperature=self.config.temperature)

    def create_plan(
        self,
        question: str,
        selected_documents: List[str],
        summaries: Dict[str, Dict],
        verbose: bool = False
    ) -> RetrievalPlan:
        """
        Create structured retrieval plan for answering the question.

        Args:
            question: User question
            selected_documents: Documents selected by Router
            summaries: Document summaries for context
            verbose: Print planning process

        Returns:
            RetrievalPlan with ordered steps
        """
        if verbose:
            print(f"\n{'=' * 60}")
            print("PLANNER AGENT - STRATEGY FORMULATION")
            print(f"{'=' * 60}")
            print(f"Question: {question}")
            print(f"Selected documents: {len(selected_documents)}")
            for doc in selected_documents:
                doc_type = summaries.get(doc, {}).get('document_type', 'unknown')
                print(f"  - {doc} ({doc_type})")

        # Build planning prompt
        prompt = self._build_planning_prompt(question, selected_documents, summaries)

        if verbose:
            print(f"\nFormulating strategy...")

        # Call LLM to create plan
        response = self._call_llm(
            prompt=prompt,
            system_prompt=self._get_system_prompt(),
            max_tokens=self.config.max_tokens,
            reset_history=True
        )

        # Parse response into structured plan
        plan = self._parse_plan(response, question, selected_documents)

        if verbose:
            print(f"\n✓ Plan created:")
            print(f"  Strategy: {plan.strategy}")
            print(f"  Steps: {len(plan.steps)}")
            for step in plan.steps:
                print(f"    {step.step_number}. {step.description}")
                print(f"       Query: \"{step.query}\"")
                print(f"       Docs: {', '.join(step.target_documents)}")

        return plan

    def _get_system_prompt(self) -> str:
        """Get system prompt for planner agent."""
        return """You are an expert planning agent for a RAG question-answering system.

Your task: Create a structured, step-by-step retrieval plan to answer complex questions.

You will be provided:
1. A question from the user
2. A list of relevant documents (already selected by Router)
3. Summaries of those documents

You must:
1. Analyze the question to understand what information is needed
2. Break down the question into logical retrieval steps
3. For each step, specify:
   - What information to retrieve
   - Which document(s) to search
   - The specific query to use
   - What the expected output is
4. Determine if steps need to be combined (e.g., multi-hop reasoning)

Return your plan as JSON in this format:
{
  "strategy": "High-level approach description",
  "steps": [
    {
      "step_number": 1,
      "description": "What this step does",
      "target_documents": ["doc1.pdf"],
      "query": "Specific search query for this step",
      "expected_output": "What information this step should find"
    }
  ],
  "success_criteria": "How to know if the plan succeeded",
  "requires_combination": true/false
}

Guidelines:
- Simple questions: 1 step (direct retrieval)
- Complex questions: Multiple steps (e.g., lookup value, then use in calculation)
- Multi-document: Steps may target different documents
- Multi-hop: Set requires_combination=true if steps must be combined
- Be specific: Clear queries that will retrieve the right information
- Be strategic: Order steps logically (dependencies first)

SPECIAL HANDLING FOR LIST/ENUMERATION QUESTIONS:

When the question asks to "list all", "list the", "enumerate", "what are all", or "show all":
1. This is an ENUMERATION task - user wants a complete list, not detailed analysis
2. Strategy: Single-step retrieval focusing on comprehensive lists or table of contents
3. Query should target: "table of contents", "list of all", "index", "complete list", "section [X]"
4. Do NOT break into multiple specific sub-category steps
5. Set requires_combination: false (simple list, no synthesis needed)
6. Keep query broad to capture the entire enumeration

Example - CORRECT approach for "List all rating plan rules":
{
  "strategy": "Retrieve comprehensive list of rating plan rules from table of contents",
  "steps": [{
    "step_number": 1,
    "description": "Retrieve complete list of all rating plan rules",
    "target_documents": ["Homeowner Rules Manual.pdf"],
    "query": "table of contents section C rating plan rules complete list",
    "expected_output": "Complete enumeration of all rating plan rules (e.g., C-1, C-2, ..., C-35)"
  }],
  "requires_combination": false
}

SPECIAL HANDLING FOR CALCULATION/PREMIUM QUESTIONS:

When the question asks to "calculate", "compute", or involves premium/rate computation:
1. This is a MULTI-STEP CALCULATION task - break into distinct value lookups
2. Each step should retrieve ONE specific input value needed for the formula
3. Use targeted queries that include the exact scenario parameters (e.g., coverage amount, distance, deductible type)
4. Set requires_combination: true (values must be combined in final calculation)
5. Order steps so dependencies come first (base rate before applying factors)

Example - CORRECT approach for "Calculate premium for HO3 with $750K coverage and 2% hurricane deductible":
{
  "strategy": "Look up each rate component separately, then combine for calculation",
  "steps": [
    {
      "step_number": 1,
      "description": "Look up the base rate for HO3 policy with $750,000 Coverage A",
      "target_documents": ["Rate Pages.pdf"],
      "query": "base rate HO3 $750,000 Coverage A limit",
      "expected_output": "Dollar amount for the base rate at this coverage level"
    },
    {
      "step_number": 2,
      "description": "Look up the Mandatory Hurricane Deductible Factor for 2% deductible",
      "target_documents": ["Rate Pages.pdf"],
      "query": "mandatory hurricane deductible factor 2 percent coastline",
      "expected_output": "Numeric multiplier for the hurricane deductible factor"
    },
    {
      "step_number": 3,
      "description": "Look up distance to coast factor for 3000 feet coastline neighborhood",
      "target_documents": ["Rate Pages.pdf"],
      "query": "distance to coast factor 3000 feet coastline neighborhood",
      "expected_output": "Factor or confirmation that distance is within coastline threshold"
    }
  ],
  "success_criteria": "All numeric values retrieved; ready to multiply base rate by applicable factors",
  "requires_combination": true
}

Key principle: Each retrieval query should target a SPECIFIC value with scenario-relevant parameters.
Avoid generic queries like "rates" or "factors" - be precise about what scenario you are looking up.

Return ONLY the JSON object, no other text."""

    def _build_planning_prompt(
        self,
        question: str,
        selected_documents: List[str],
        summaries: Dict[str, Dict]
    ) -> str:
        """
        Build prompt for plan creation.

        Args:
            question: User question
            selected_documents: Documents to search
            summaries: Document summaries

        Returns:
            Formatted prompt string
        """
        # Format document information
        doc_info = []
        for doc in selected_documents:
            summary_data = summaries.get(doc, {})
            info = f"""
Document: {doc}
  Type: {summary_data.get('document_type', 'unknown')}
  Summary: {summary_data.get('summary', 'N/A')}
  Use for: {summary_data.get('use_for', 'N/A')}
  Key topics: {', '.join(summary_data.get('key_topics', [])[:5])}
"""
            doc_info.append(info.strip())

        documents_text = "\n\n".join(doc_info)

        # Detect enumeration/list questions
        question_lower = question.lower()
        enumeration_patterns = [
            "list all", "list the", "what are all", "enumerate",
            "show all", "give me all", "what are the"
        ]
        is_enumeration = any(pattern in question_lower for pattern in enumeration_patterns)

        # Detect calculation questions
        calculation_patterns = [
            "calculate", "compute", "what is the premium", "how much"
        ]
        is_calculation = any(pattern in question_lower for pattern in calculation_patterns)

        # Add contextual hints based on question type
        type_hint = ""
        if is_enumeration:
            type_hint = """
⚠️  ENUMERATION TASK DETECTED
This question asks for a complete list/enumeration.
- Use single-step retrieval focused on table of contents, index, or comprehensive lists
- Query should be broad: "table of contents [topic]", "complete list of [items]", "all [items]"
- Do NOT break into specific sub-categories (will miss items)
- Set requires_combination: false
"""
        elif is_calculation:
            type_hint = """
⚠️  CALCULATION TASK DETECTED
This question requires extracting specific numeric values and performing a computation.
- Break into multiple steps, one per input value (base rate, factor, deductible, etc.)
- Each query should target a SPECIFIC value with scenario-relevant parameters from the question
- Include numeric thresholds from the question (e.g., coverage amount, distance, deductible percentage)
- Set requires_combination: true (values must be combined for the final calculation)
- Order steps by dependency: base values first, then modifiers
"""

        prompt = f"""Question: "{question}"
{type_hint}
Available documents (selected by Router):
{documents_text}

Analyze this question and create a structured retrieval plan.

Consider:
- Is this a simple lookup or complex multi-step question?
- Does it require information from multiple documents?
- Are there dependencies between steps?
- Does it involve calculations or combinations of information?
- For LIST/ENUMERATION questions: Focus on comprehensive retrieval from indexes/TOCs
- For CALCULATION questions: Each step should look up one specific numeric input

Create a detailed retrieval plan as JSON."""

        return prompt

    def _parse_plan(
        self,
        response: str,
        question: str,
        selected_documents: List[str]
    ) -> RetrievalPlan:
        """
        Parse LLM response into structured RetrievalPlan.

        Args:
            response: Raw LLM response
            question: Original question
            selected_documents: Available documents

        Returns:
            RetrievalPlan object
        """
        try:
            # Extract JSON from response
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()

            # Parse JSON
            plan_data = json.loads(response)

            # Validate required fields
            required = ["strategy", "steps", "success_criteria"]
            if not all(k in plan_data for k in required):
                raise ValueError(f"Missing required fields. Got: {plan_data.keys()}")

            # Parse steps
            steps = []
            for step_data in plan_data["steps"]:
                step = RetrievalStep(
                    step_number=step_data["step_number"],
                    description=step_data["description"],
                    target_documents=step_data["target_documents"],
                    query=step_data["query"],
                    expected_output=step_data["expected_output"]
                )
                steps.append(step)

            # Create plan
            plan = RetrievalPlan(
                question=question,
                strategy=plan_data["strategy"],
                steps=steps,
                success_criteria=plan_data["success_criteria"],
                requires_combination=plan_data.get("requires_combination", False)
            )

            return plan

        except Exception as e:
            print(f"  ⚠️  Failed to parse plan: {e}")
            print(f"     Raw response: {response[:300]}...")

            # Fallback: create simple single-step plan
            print(f"     Using fallback: single-step plan")
            return RetrievalPlan(
                question=question,
                strategy="Direct retrieval from selected documents",
                steps=[
                    RetrievalStep(
                        step_number=1,
                        description="Search all selected documents for relevant information",
                        target_documents=selected_documents,
                        query=question,
                        expected_output="Information relevant to the question"
                    )
                ],
                success_criteria="Find relevant information in documents",
                requires_combination=False
            )

    def refine_plan(
        self,
        plan: RetrievalPlan,
        feedback: str,
        verbose: bool = False
    ) -> RetrievalPlan:
        """
        Refine plan based on feedback (conversational refinement).

        Args:
            plan: Existing plan to refine
            feedback: Feedback about the plan (e.g., "Step 2 should search rates.pdf not rules.pdf")
            verbose: Print refinement process

        Returns:
            Refined RetrievalPlan
        """
        if verbose:
            print(f"\n{'─' * 60}")
            print("REFINING PLAN")
            print(f"{'─' * 60}")
            print(f"Feedback: {feedback}")

        # Build refinement prompt
        plan_json = json.dumps(asdict(plan), indent=2)

        prompt = f"""Current plan:
{plan_json}

Feedback: {feedback}

Refine the plan based on this feedback.
Return the updated plan as JSON in the same format."""

        # Call LLM (conversation history preserved)
        response = self._call_llm(
            prompt=prompt,
            max_tokens=self.config.max_tokens
        )

        # Parse refined plan
        refined_plan = self._parse_plan(
            response,
            plan.question,
            [doc for step in plan.steps for doc in step.target_documents]
        )

        if verbose:
            print(f"✓ Plan refined")

        return refined_plan

    def to_dict(self, plan: RetrievalPlan) -> Dict:
        """Convert plan to dictionary for serialization."""
        return asdict(plan)

    def from_dict(self, plan_dict: Dict) -> RetrievalPlan:
        """Load plan from dictionary."""
        steps = [RetrievalStep(**step) for step in plan_dict["steps"]]
        return RetrievalPlan(
            question=plan_dict["question"],
            strategy=plan_dict["strategy"],
            steps=steps,
            success_criteria=plan_dict["success_criteria"],
            requires_combination=plan_dict.get("requires_combination", False)
        )

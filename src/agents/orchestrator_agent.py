"""
Orchestrator Agent for multi-agent RAG system.

Coordinates Router, Planner, and Retriever agents to answer questions about PDFs.
"""
from typing import Optional, Dict, List
from dataclasses import dataclass

from .base_agent import BaseAgent
from .router_agent import RouterAgent, RouterConfig
from .planner_agent import PlannerAgent, PlannerConfig
from .retriever_agent import RetrieverAgent, RetrieverConfig, ExecutionResult


@dataclass
class OrchestratorConfig:
    """Configuration for Orchestrator Agent."""
    model: str = "llama3.2"
    temperature: float = 0.0
    max_answer_tokens: int = 2048
    router_config: Optional[RouterConfig] = None
    planner_config: Optional[PlannerConfig] = None
    retriever_config: Optional[RetrieverConfig] = None


@dataclass
class Answer:
    """Structured answer with sources."""
    question: str
    answer: str
    sources: List[Dict[str, str]]  # List of {file, page, text} dicts
    retrieved_chunks: int
    steps_executed: int


class OrchestratorAgent(BaseAgent):
    """
    Orchestrator agent that coordinates the full RAG pipeline.

    Pipeline: Router → Planner → Retriever → Answer Generation

    Example:
        orchestrator = OrchestratorAgent(
            summaries_path="artifacts/document_summaries.json",
            pdf_folder="artifacts/1"
        )
        answer = orchestrator.answer_question(
            question="What are the rules for an ineligible risk?"
        )
        print(answer.answer)
    """

    def __init__(
        self,
        summaries_path: str,
        pdf_folder: str,
        config: Optional[OrchestratorConfig] = None
    ):
        """
        Initialize Orchestrator Agent.

        Args:
            summaries_path: Path to document_summaries.json
            pdf_folder: Path to folder containing PDFs
            config: Orchestrator configuration
        """
        self.config = config or OrchestratorConfig()
        super().__init__(
            model=self.config.model,
            temperature=self.config.temperature
        )

        # Initialize sub-agents
        self.router = RouterAgent(
            summaries_path=summaries_path,
            config=self.config.router_config or RouterConfig()
        )

        self.planner = PlannerAgent(
            config=self.config.planner_config or PlannerConfig()
        )

        self.retriever = RetrieverAgent(
            pdf_folder=pdf_folder,
            config=self.config.retriever_config or RetrieverConfig()
        )

        print(f"✓ Orchestrator initialized with {self.config.model}")

    def answer_question(
        self,
        question: str,
        verbose: bool = False
    ) -> Answer:
        """
        Answer a question using the full RAG pipeline.

        Args:
            question: User question
            verbose: Print intermediate steps

        Returns:
            Answer object with answer text and sources
        """
        if verbose:
            print(f"\n{'=' * 70}")
            print("ORCHESTRATOR - FULL RAG PIPELINE")
            print(f"{'=' * 70}")
            print(f"Question: {question}\n")

        # Step 1: Route to relevant documents
        if verbose:
            print(f"{'─' * 70}")
            print("[STEP 1] ROUTER - Document Selection")
            print(f"{'─' * 70}")

        selected_docs = self.router.select_documents(
            question=question,
            verbose=verbose
        )

        if verbose:
            print(f"\n✓ Selected {len(selected_docs)} documents\n")

        # Step 2: Create retrieval plan
        if verbose:
            print(f"{'─' * 70}")
            print("[STEP 2] PLANNER - Strategy Formulation")
            print(f"{'─' * 70}")

        plan = self.planner.create_plan(
            question=question,
            selected_documents=selected_docs,
            summaries=self.router.summaries,  # Pass summaries from router
            verbose=verbose
        )

        if verbose:
            print(f"\n✓ Created plan with {len(plan.steps)} steps\n")

        # Step 3: Execute retrieval plan
        if verbose:
            print(f"{'─' * 70}")
            print("[STEP 3] RETRIEVER - Plan Execution")
            print(f"{'─' * 70}")

        execution_result = self.retriever.execute_plan(
            plan=plan,
            verbose=verbose
        )

        total_chunks = sum(len(r.chunks) for r in execution_result.step_results)

        if verbose:
            print(f"\n✓ Retrieved {total_chunks} total chunks across {len(plan.steps)} steps\n")

        # Step 4: Generate answer from retrieved chunks
        if verbose:
            print(f"{'─' * 70}")
            print("[STEP 4] ANSWER GENERATION")
            print(f"{'─' * 70}")

        answer = self._generate_answer(
            question=question,
            execution_result=execution_result,
            verbose=verbose
        )

        if verbose:
            print(f"\n✓ Answer generated ({len(answer.answer)} characters)\n")
            print(f"{'=' * 70}")
            print("FINAL ANSWER")
            print(f"{'=' * 70}")
            print(answer.answer)
            print(f"\n{'─' * 70}")
            print(f"Sources: {len(answer.sources)} chunks from {len(set(s['file'] for s in answer.sources))} documents")
            print(f"{'─' * 70}\n")

        return answer

    def _generate_answer(
        self,
        question: str,
        execution_result: ExecutionResult,
        verbose: bool = False
    ) -> Answer:
        """
        Generate final answer from retrieved chunks.

        Args:
            question: User question
            execution_result: Result from Retriever
            verbose: Print details

        Returns:
            Answer object
        """
        # Build context from retrieved chunks
        context = self._format_context(execution_result)

        # Build prompt
        prompt = self._build_answer_prompt(question, context, execution_result)

        if verbose:
            print(f"  Context: {len(context)} characters from {sum(len(r.chunks) for r in execution_result.step_results)} chunks")
            print(f"  Generating answer...")

        # Generate answer
        answer_text = self._call_llm(
            prompt=prompt,
            system_prompt=self._get_answer_system_prompt(),
            max_tokens=self.config.max_answer_tokens,
            reset_history=True
        )

        # Extract sources
        sources = self._extract_sources(execution_result)

        return Answer(
            question=question,
            answer=answer_text,
            sources=sources,
            retrieved_chunks=sum(len(r.chunks) for r in execution_result.step_results),
            steps_executed=len(execution_result.step_results)
        )

    def _format_context(self, execution_result: ExecutionResult) -> str:
        """
        Format retrieved chunks as context for answer generation.

        Args:
            execution_result: Result from Retriever

        Returns:
            Formatted context string
        """
        context_parts = []

        for step_result in execution_result.step_results:
            if not step_result.chunks:
                continue

            # Add step header
            step_header = f"=== Step {step_result.step_number}: {step_result.description} ==="
            context_parts.append(step_header)
            context_parts.append("")

            # Add chunks
            for i, chunk in enumerate(step_result.chunks, 1):
                source_file = chunk['source_file']
                page = chunk['page_number']
                text = chunk['text']

                chunk_text = f"[{i}] {source_file} (page {page}):\n{text}"
                context_parts.append(chunk_text)
                context_parts.append("")

        return "\n".join(context_parts)

    def _build_answer_prompt(
        self,
        question: str,
        context: str,
        execution_result: ExecutionResult
    ) -> str:
        """
        Build prompt for answer generation.

        Args:
            question: User question
            context: Formatted context from retrieved chunks
            execution_result: Execution result with plan info

        Returns:
            Formatted prompt
        """
        prompt = f"""Question: "{question}"

Strategy: {execution_result.plan.strategy}

Retrieved Information:
{context}

Instructions:
1. Answer the question using ONLY the information provided above
2. Be specific and cite relevant details from the documents
3. If the information is insufficient or unclear, state what is missing
4. Do not make assumptions beyond what is stated in the documents
5. Structure your answer clearly (use bullet points if appropriate)

Answer:"""

        return prompt

    def _get_answer_system_prompt(self) -> str:
        """Get system prompt for answer generation."""
        return """You are an expert insurance underwriting and rating analyst.

Your task: Answer questions accurately based on provided document excerpts.

Guidelines:
- Use ONLY information from the provided context
- Be precise and specific
- Cite relevant details (document names, page numbers, specific values)
- If information is incomplete, clearly state what is missing
- Do not hallucinate or make assumptions
- Structure complex answers with bullet points or numbered lists
- Use professional, clear language"""

    def _extract_sources(self, execution_result: ExecutionResult) -> List[Dict[str, str]]:
        """
        Extract source citations from execution result.

        Args:
            execution_result: Result from Retriever

        Returns:
            List of source dicts with file, page, and text preview
        """
        sources = []

        for step_result in execution_result.step_results:
            for chunk in step_result.chunks:
                source = {
                    'file': chunk['source_file'],
                    'page': str(chunk['page_number']),
                    'text': chunk['text'][:200] + '...' if len(chunk['text']) > 200 else chunk['text']
                }
                sources.append(source)

        return sources

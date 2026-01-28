"""
Retriever Agent for executing multi-step retrieval plans.

Executes structured plans using existing RAG processors.
"""
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path

from .base_agent import BaseAgent
from .planner_agent import RetrievalPlan, RetrievalStep
from ..onnx_processor import ONNXProcessor
from ..ollama_processor import OllamaProcessor
from ..hybrid_retriever import HybridRetriever


@dataclass
class RetrievalResult:
    """Result from executing a single retrieval step."""
    step_number: int
    description: str  # Step description
    query: str
    chunks: List[Dict[str, Any]]  # Retrieved chunks with metadata
    num_chunks: int
    target_documents: List[str]


@dataclass
class ExecutionResult:
    """Complete result from executing a retrieval plan."""
    question: str
    plan_strategy: str
    step_results: List[RetrievalResult]
    requires_combination: bool
    combined_context: Optional[str] = None  # If combination needed


@dataclass
class RetrieverConfig:
    """Configuration for Retriever Agent."""
    embedding_type: str = "onnx"  # "onnx" or "ollama"
    top_k_per_step: int = 5  # Chunks per step
    chunk_size: int = 1000  # Characters per chunk
    chunk_overlap: int = 200  # Character overlap between chunks
    use_hybrid: bool = True  # Enable hybrid BM25 + semantic search
    hybrid_alpha: float = 0.5  # Weight for semantic (1-alpha for BM25)
    model: str = "llama3.2"  # For combination/synthesis
    temperature: float = 0.0


class RetrieverAgent(BaseAgent):
    """
    Retrieval execution agent.

    Executes multi-step retrieval plans using existing processors.
    Combines results if required for multi-hop reasoning.

    Example:
        retriever = RetrieverAgent(
            processor=onnx_processor,
            config=RetrieverConfig(embedding_type="onnx")
        )

        result = retriever.execute_plan(plan, verbose=True)
    """

    def __init__(
        self,
        processor: Optional[Any] = None,  # ONNXProcessor or OllamaProcessor
        config: Optional[RetrieverConfig] = None,
        pdf_folder: str = "artifacts/1"
    ):
        """
        Initialize Retriever Agent.

        Args:
            processor: Existing processor instance (optional, will create if not provided)
            config: Retriever configuration
            pdf_folder: Path to PDF folder (for processor initialization)
        """
        self.config = config or RetrieverConfig()
        super().__init__(model=self.config.model, temperature=self.config.temperature)

        # Store pdf_folder for potential indexing
        self.pdf_folder = pdf_folder

        # Set up processor
        if processor is None:
            self.processor = self._create_processor(pdf_folder)
        else:
            self.processor = processor
            # Check if processor has data, index if empty
            if self.processor.count() == 0:
                print(f"⚠️  Processor has no data. Indexing PDFs from {pdf_folder}...")
                self.processor.process_folder(pdf_folder)
                print(f"✓ Indexed {self.processor.count()} chunks")

        # Initialize hybrid retriever if enabled
        self.hybrid_retriever = None
        if self.config.use_hybrid:
            print(f"Building hybrid (BM25 + semantic) index...")
            self.hybrid_retriever = HybridRetriever(self.processor)
            self.hybrid_retriever.build_bm25_index(verbose=True)
            print(f"✓ Retriever initialized with HYBRID search ({self.processor.count()} chunks)")
        else:
            print(f"✓ Retriever initialized with {self.config.embedding_type} embeddings ({self.processor.count()} chunks)")

    def _create_processor(self, pdf_folder: str):
        """
        Create processor if not provided.

        Args:
            pdf_folder: Path to PDF folder

        Returns:
            Processor instance
        """
        # Generate unique collection name based on configuration
        collection_name = f"pdf_{self.config.embedding_type}_{self.config.chunk_size}_{self.config.chunk_overlap}"

        if self.config.embedding_type == "onnx":
            processor = ONNXProcessor(
                persist_directory="./chroma_db_onnx",
                collection_name=collection_name,
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
            )
        elif self.config.embedding_type == "ollama":
            processor = OllamaProcessor(
                persist_directory="./chroma_db_ollama",
                collection_name=collection_name,
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
            )
        else:
            raise ValueError(f"Unknown embedding type: {self.config.embedding_type}")

        # Index if needed
        if processor.count() == 0:
            print(f"Indexing PDFs from {pdf_folder}...")
            processor.process_folder(pdf_folder)

        return processor

    def execute_plan(
        self,
        plan: RetrievalPlan,
        verbose: bool = False
    ) -> ExecutionResult:
        """
        Execute retrieval plan step by step.

        Args:
            plan: RetrievalPlan from Planner agent
            verbose: Print execution details

        Returns:
            ExecutionResult with retrieved information
        """
        if verbose:
            print(f"\n{'=' * 60}")
            print("RETRIEVER AGENT - PLAN EXECUTION")
            print(f"{'=' * 60}")
            print(f"Question: {plan.question}")
            print(f"Strategy: {plan.strategy}")
            print(f"Steps: {len(plan.steps)}")
            print(f"Requires combination: {plan.requires_combination}")

        # Execute each step
        step_results = []
        for step in plan.steps:
            if verbose:
                print(f"\n{'─' * 60}")
                print(f"Executing Step {step.step_number}: {step.description}")
                print(f"{'─' * 60}")
                print(f"  Query: \"{step.query}\"")
                print(f"  Target docs: {', '.join(step.target_documents)}")

            result = self._execute_step(step, verbose=verbose)
            step_results.append(result)

            if verbose:
                print(f"  ✓ Retrieved {result.num_chunks} chunks")

        # Combine results if needed
        combined_context = None
        if plan.requires_combination:
            if verbose:
                print(f"\n{'─' * 60}")
                print("Combining multi-step results...")
                print(f"{'─' * 60}")

            combined_context = self._combine_results(step_results, plan.question, verbose=verbose)

            if verbose:
                print(f"  ✓ Context combined ({len(combined_context)} chars)")

        # Create execution result
        execution_result = ExecutionResult(
            question=plan.question,
            plan_strategy=plan.strategy,
            step_results=step_results,
            requires_combination=plan.requires_combination,
            combined_context=combined_context
        )

        if verbose:
            print(f"\n{'=' * 60}")
            print("✓ Plan execution complete")
            print(f"{'=' * 60}")

        return execution_result

    def _execute_step(
        self,
        step: RetrievalStep,
        verbose: bool = False
    ) -> RetrievalResult:
        """
        Execute a single retrieval step.

        Uses pre-filtered search: only searches within target documents specified in the plan.
        Falls back to search-then-filter if pre-filtering fails.

        Args:
            step: RetrievalStep to execute
            verbose: Print details

        Returns:
            RetrievalResult with retrieved chunks
        """
        # Detect enumeration queries (favor BM25 keyword matching)
        query_lower = step.query.lower()
        is_enumeration = any(term in query_lower for term in [
            "list", "all", "table of contents", "index", "enumerate",
            "rules", "complete", "comprehensive"
        ])

        # Determine search strategy
        used_prefilter = False
        where_clause = None

        if step.target_documents:
            where_clause = {"source_file": {"$in": step.target_documents}}

        # Execute search (hybrid or semantic)
        if self.config.use_hybrid and self.hybrid_retriever:
            # Hybrid search: adjust alpha based on query type
            alpha = self.config.hybrid_alpha
            if is_enumeration:
                # Favor BM25 for enumeration (lower alpha)
                alpha = 0.3  # 70% BM25, 30% semantic
                search_type = "HYBRID (BM25-heavy)"
            else:
                # Balanced or semantic-heavy for reasoning
                alpha = 0.7  # 30% BM25, 70% semantic
                search_type = "HYBRID (semantic-heavy)"

            if verbose:
                print(f"    Using {search_type} search (alpha={alpha:.1f})")

            try:
                results = self.hybrid_retriever.search(
                    query=step.query,
                    top_k=self.config.top_k_per_step,
                    alpha=alpha,
                    where=where_clause,
                    verbose=verbose
                )
                used_prefilter = where_clause is not None

            except Exception as e:
                if verbose:
                    print(f"    ⚠️  Hybrid search failed: {e}")
                    print(f"    Falling back to semantic-only...")
                # Fallback to semantic
                results = self.processor.query(
                    query_text=step.query,
                    top_k=self.config.top_k_per_step
                )
                used_prefilter = False

        elif step.target_documents:
            try:
                # Use ChromaDB's where clause to pre-filter by source_file
                results = self.processor.collection.query(
                    query_texts=[step.query],
                    n_results=self.config.top_k_per_step,
                    where=where_clause
                )
                used_prefilter = True

                if verbose:
                    print(f"    Using pre-filtered search (target docs only)")

            except Exception as e:
                # Fallback to search-then-filter if pre-filtering fails
                if verbose:
                    print(f"    ⚠️  Pre-filtered search failed: {e}")
                    print(f"    Falling back to search-then-filter...")

                results = self.processor.query(
                    query_text=step.query,
                    top_k=self.config.top_k_per_step
                )
                used_prefilter = False
        else:
            # No target documents specified, search everything
            results = self.processor.query(
                query_text=step.query,
                top_k=self.config.top_k_per_step
            )
            used_prefilter = False

        # Extract chunks with metadata
        chunks = []
        filtered_out = 0

        if results and results.get('documents') and results['documents'][0]:
            for i, (doc_text, metadata) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0]
            )):
                source_file = metadata.get('source_file', '')

                # If pre-filtering was used successfully, trust all results
                # Otherwise (fallback mode), filter by target documents
                if not used_prefilter and step.target_documents and source_file not in step.target_documents:
                    filtered_out += 1
                    continue

                chunk = {
                    'text': doc_text,
                    'source_file': source_file,
                    'page_number': metadata.get('page_number', 'unknown'),
                    'chunk_index': metadata.get('chunk_index', i),
                    'distance': results.get('distances', [[]])[0][i] if results.get('distances') else None
                }
                chunks.append(chunk)

                if verbose and len(chunks) <= 3:  # Show first 3
                    print(f"    [{i+1}] {source_file} (p{chunk['page_number']}): {doc_text[:80]}...")

        # Fallback to partial matching if we're in search-then-filter mode and all chunks were filtered out
        # (Not needed if pre-filtering was successful, since pre-filter guarantees correct documents)
        if not used_prefilter and len(chunks) == 0 and filtered_out > 0 and step.target_documents:
            if verbose:
                print(f"    ⚠️  All {filtered_out} chunks filtered out by exact match")
                print(f"    Attempting partial filename matching...")

            # Try partial matching
            for i, (doc_text, metadata) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0]
            )):
                source_file = metadata.get('source_file', '')

                # Check if any target document is a substring of source_file or vice versa
                match_found = False
                matched_target = None
                for target_doc in step.target_documents:
                    # Remove common prefixes like "(numbers-numbers)-" for comparison
                    clean_source = source_file.split(')-', 1)[-1] if ')-' in source_file else source_file
                    clean_target = target_doc.split(')-', 1)[-1] if ')-' in target_doc else target_doc

                    if clean_target in clean_source or clean_source in clean_target:
                        match_found = True
                        matched_target = target_doc
                        break

                if not match_found:
                    continue

                chunk = {
                    'text': doc_text,
                    'source_file': source_file,
                    'page_number': metadata.get('page_number', 'unknown'),
                    'chunk_index': metadata.get('chunk_index', i),
                    'distance': results.get('distances', [[]])[0][i] if results.get('distances') else None
                }
                chunks.append(chunk)

                if verbose and len(chunks) <= 3:  # Show first 3
                    print(f"    [{i+1}] {source_file} (p{chunk['page_number']}) [partial match: {matched_target}]")
                    print(f"        {doc_text[:80]}...")

            if verbose:
                if len(chunks) > 0:
                    print(f"    ✓ Partial matching retrieved {len(chunks)} chunks")
                else:
                    print(f"    ⚠️  No matches found even with partial matching")

        # Diagnostic warnings for debugging
        if verbose and len(chunks) == 0:
            if used_prefilter:
                print(f"    ⚠️  Pre-filtered search returned 0 chunks")
                print(f"    Target docs: {step.target_documents}")
                print(f"    This may indicate the target documents don't contain relevant content")
            elif filtered_out > 0:
                print(f"    ⚠️  All {filtered_out} chunks filtered out (document name mismatch)")
                print(f"    Target docs: {step.target_documents}")
                if results and results.get('metadatas') and results['metadatas'][0]:
                    unique_sources = set(m.get('source_file', '') for m in results['metadatas'][0][:3])
                    print(f"    Retrieved from: {list(unique_sources)}")

        return RetrievalResult(
            step_number=step.step_number,
            description=step.description,
            query=step.query,
            chunks=chunks,
            num_chunks=len(chunks),
            target_documents=step.target_documents
        )

    def _combine_results(
        self,
        step_results: List[RetrievalResult],
        question: str,
        verbose: bool = False
    ) -> str:
        """
        Combine results from multiple steps (multi-hop reasoning).

        Args:
            step_results: Results from all steps
            question: Original question
            verbose: Print combination process

        Returns:
            Combined context string
        """
        # Build context from all steps
        context_parts = []

        for result in step_results:
            step_context = f"=== Step {result.step_number} ({result.query}) ===\n"

            for i, chunk in enumerate(result.chunks[:3], 1):  # Top 3 per step
                step_context += f"\n[{i}] {chunk['source_file']} (page {chunk['page_number']}):\n"
                step_context += f"{chunk['text'][:500]}...\n"

            context_parts.append(step_context)

        # Combine all contexts
        combined = "\n\n".join(context_parts)

        # Optionally use LLM to synthesize (for now, just concatenate)
        # Future: Could ask LLM to synthesize/summarize the combined context

        if verbose:
            print(f"  Combined context from {len(step_results)} steps")

        return combined

    def get_answer_context(self, execution_result: ExecutionResult, max_chunks: int = 5) -> str:
        """
        Get formatted context for answer generation.

        Args:
            execution_result: Result from execute_plan
            max_chunks: Maximum chunks to include

        Returns:
            Formatted context string for LLM
        """
        if execution_result.combined_context:
            return execution_result.combined_context

        # No combination needed - format step results
        context_parts = []

        for result in execution_result.step_results:
            for chunk in result.chunks[:max_chunks]:
                context_parts.append(
                    f"[{chunk['source_file']}, page {chunk['page_number']}]\n{chunk['text']}"
                )

        return "\n\n---\n\n".join(context_parts)

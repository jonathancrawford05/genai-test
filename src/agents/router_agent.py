"""
Router Agent for document selection in multi-agent RAG system.

Selects the most relevant documents from summaries based on a question.
"""
import json
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

from .base_agent import BaseAgent


@dataclass
class RouterConfig:
    """Configuration for Router Agent."""
    model: str = "llama3.2"
    top_k_docs: int = 3
    temperature: float = 0.0
    max_tokens: int = 512


class RouterAgent(BaseAgent):
    """
    Document routing agent.

    Selects top-k most relevant documents from summaries to answer a question.

    Example:
        router = RouterAgent(summaries_path="artifacts/document_summaries.json")
        docs = router.select_documents(
            question="What are the rules for ineligible risks?",
            top_k=3
        )
        # Returns: ["CT_Rules_Manual.pdf", "Underwriting_Guide.pdf", ...]
    """

    def __init__(
        self,
        summaries_path: str = "artifacts/document_summaries.json",
        config: Optional[RouterConfig] = None
    ):
        """
        Initialize Router Agent.

        Args:
            summaries_path: Path to document summaries JSON file
            config: Router configuration (uses defaults if not provided)
        """
        self.config = config or RouterConfig()
        super().__init__(model=self.config.model, temperature=self.config.temperature)

        # Load document summaries
        self.summaries = self._load_summaries(summaries_path)
        print(f"✓ Router loaded {len(self.summaries)} document summaries")

    def _load_summaries(self, summaries_path: str) -> Dict[str, Dict]:
        """
        Load document summaries from JSON file.

        Args:
            summaries_path: Path to summaries JSON

        Returns:
            Dictionary mapping filenames to summary data
        """
        path = Path(summaries_path)
        if not path.exists():
            raise FileNotFoundError(f"Summaries file not found: {summaries_path}")

        with open(path, 'r', encoding='utf-8') as f:
            summaries = json.load(f)

        return summaries

    def select_documents(
        self,
        question: str,
        top_k: Optional[int] = None,
        verbose: bool = False
    ) -> List[str]:
        """
        Select top-k most relevant documents for answering the question.

        Args:
            question: User question
            top_k: Number of documents to select (uses config default if not provided)
            verbose: Print reasoning if True

        Returns:
            List of document filenames (sorted by relevance)
        """
        top_k = top_k or self.config.top_k_docs

        # Build prompt
        prompt = self._build_routing_prompt(question, top_k)

        if verbose:
            print(f"\n{'=' * 60}")
            print("ROUTER AGENT - DOCUMENT SELECTION")
            print(f"{'=' * 60}")
            print(f"Question: {question}")
            print(f"Selecting top {top_k} documents from {len(self.summaries)}...")

        # Call LLM
        response = self._call_llm(
            prompt=prompt,
            system_prompt=self._get_system_prompt(),
            max_tokens=self.config.max_tokens,
            reset_history=True  # Each routing is independent
        )

        # Parse response
        selected_docs = self._parse_response(response, top_k)

        if verbose:
            print(f"\nSelected documents:")
            for i, doc in enumerate(selected_docs, 1):
                doc_info = self.summaries.get(doc, {})
                doc_type = doc_info.get('document_type', 'unknown')
                print(f"  {i}. {doc} ({doc_type})")

        return selected_docs

    def _get_system_prompt(self) -> str:
        """Get system prompt for router agent."""
        return """You are an expert document routing agent for an insurance underwriting and rating system.

Your task: Select the most relevant documents to answer a given question.

You will be provided:
1. A question from a user
2. Summaries of all available documents

You must:
1. Analyze the question to understand what information is needed
2. Review each document summary to determine relevance
3. Select the top-k most relevant documents
4. Return ONLY a JSON array of filenames

Be strategic:
- Prioritize documents that directly contain the needed information
- Consider document types (rules manuals for eligibility, rate pages for pricing, etc.)
- Select complementary documents (e.g., both rules and rates if needed)
- Focus on relevance, not just keyword matching

Return format: ["filename1.pdf", "filename2.pdf", "filename3.pdf"]

Return ONLY the JSON array, no explanation or other text."""

    def _build_routing_prompt(self, question: str, top_k: int) -> str:
        """
        Build prompt for document routing.

        Args:
            question: User question
            top_k: Number of documents to select

        Returns:
            Formatted prompt string
        """
        # Format document summaries for prompt
        doc_list = []
        for i, (filename, summary_data) in enumerate(sorted(self.summaries.items()), 1):
            doc_entry = f"""
{i}. {filename}
   Type: {summary_data.get('document_type', 'unknown')}
   Summary: {summary_data.get('summary', 'No summary available')}
   Use for: {summary_data.get('use_for', 'N/A')}
   Key topics: {', '.join(summary_data.get('key_topics', [])[:5])}
"""
            doc_list.append(doc_entry.strip())

        documents_text = "\n\n".join(doc_list)

        prompt = f"""Question: "{question}"

Available documents ({len(self.summaries)} total):
{documents_text}

Select the {top_k} most relevant documents to answer this question.
Return ONLY a JSON array of filenames: ["file1.pdf", "file2.pdf", ...]"""

        return prompt

    def _parse_response(self, response: str, top_k: int) -> List[str]:
        """
        Parse LLM response to extract document filenames.

        Args:
            response: Raw LLM response
            top_k: Expected number of documents

        Returns:
            List of document filenames
        """
        try:
            # Try to extract JSON from response
            # Handle cases where LLM wraps JSON in markdown code blocks
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()

            # Parse JSON
            selected = json.loads(response)

            if not isinstance(selected, list):
                raise ValueError("Response is not a JSON array")

            # Validate filenames exist in summaries
            valid_docs = []
            for doc in selected[:top_k]:  # Take only top_k
                if doc in self.summaries:
                    valid_docs.append(doc)
                else:
                    print(f"  ⚠️  Warning: Document not found in summaries: {doc}")

            if not valid_docs:
                raise ValueError("No valid documents in response")

            return valid_docs

        except Exception as e:
            print(f"  ⚠️  Failed to parse LLM response: {e}")
            print(f"     Raw response: {response[:200]}...")

            # Fallback: return first k documents alphabetically
            print(f"     Using fallback: first {top_k} documents alphabetically")
            return sorted(self.summaries.keys())[:top_k]

    def get_document_summary(self, filename: str) -> Optional[Dict]:
        """
        Get summary for a specific document.

        Args:
            filename: Document filename

        Returns:
            Summary dictionary or None if not found
        """
        return self.summaries.get(filename)

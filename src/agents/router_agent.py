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

        Uses two-step approach:
        1. Free-form reasoning to identify candidate documents
        2. Tool calling to select exact filenames from valid list

        Args:
            question: User question
            top_k: Number of documents to select (uses config default if not provided)
            verbose: Print reasoning if True

        Returns:
            List of document filenames (sorted by relevance)
        """
        top_k = top_k or self.config.top_k_docs

        if verbose:
            print(f"\n{'=' * 60}")
            print("ROUTER AGENT - DOCUMENT SELECTION")
            print(f"{'=' * 60}")
            print(f"Question: {question}")
            print(f"Selecting top {top_k} documents from {len(self.summaries)}...")

        # Step 1: Free-form reasoning about relevant documents
        reasoning_prompt = self._build_reasoning_prompt(question, top_k)

        if verbose:
            print(f"\n[Step 1] Analyzing document relevance...")

        reasoning_response = self._call_llm(
            prompt=reasoning_prompt,
            system_prompt=self._get_reasoning_system_prompt(),
            max_tokens=1024,  # More tokens for reasoning
            reset_history=True
        )

        if verbose:
            print(f"Reasoning: {reasoning_response[:200]}...")

        # Step 2: Select exact filenames using tool calling
        selection_prompt = self._build_selection_prompt(question, reasoning_response, top_k)
        tools = self._get_selection_tools()

        if verbose:
            print(f"\n[Step 2] Selecting exact filenames with validation...")

        try:
            response = self._call_llm_with_tools(
                prompt=selection_prompt,
                tools=tools,
                system_prompt=self._get_selection_system_prompt(),
                max_tokens=512,
                reset_history=False  # Keep reasoning in context
            )

            # Parse tool call response
            selected_docs = self._parse_tool_response(response, top_k)

        except Exception as e:
            if verbose:
                print(f"  ⚠️  Tool calling failed: {e}")
                print(f"     Falling back to direct parsing...")

            # Fallback: try to parse reasoning response directly
            selected_docs = self._parse_response(reasoning_response, top_k)

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
4. Return ONLY a JSON array of EXACT filenames as shown in the document list

Be strategic:
- Prioritize documents that directly contain the needed information
- Consider document types (rules manuals for eligibility, rate pages for pricing, etc.)
- Select complementary documents (e.g., both rules and rates if needed)
- Focus on relevance, not just keyword matching

CRITICAL: Return the EXACT filename as shown in the document list, including all prefixes, ID numbers, and version numbers.

CORRECT example: ["(215066178-180449588)-CT MAPS Homeowner Rules Manual eff 08.18.25 v4.pdf"]
WRONG example: ["CT MAPS Homeowner Rules Manual eff 08.18.25 v4.pdf"]

Return format: ["exact_filename1.pdf", "exact_filename2.pdf", "exact_filename3.pdf"]

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

IMPORTANT: Copy the EXACT filename from the list above, including all prefixes like "(12345-67890)-" and version numbers.

Return ONLY a JSON array of EXACT filenames: ["exact_file1.pdf", "exact_file2.pdf", ...]"""

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

    def _build_reasoning_prompt(self, question: str, top_k: int) -> str:
        """
        Build prompt for reasoning step (identifies candidate documents).

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

Analyze which {top_k} documents would be most relevant to answer this question.

Consider:
- What type of information does the question need? (rules, rates, calculations, etc.)
- Which documents contain that type of information?
- Are multiple documents needed for a complete answer?

List the top {top_k} candidate documents with brief reasoning for each."""

        return prompt

    def _get_reasoning_system_prompt(self) -> str:
        """Get system prompt for reasoning step."""
        return """You are an expert document analyst for an insurance underwriting and rating system.

Your task: Analyze which documents are most relevant to answer a given question.

Think step-by-step:
1. What type of information does the question require?
2. Which document types typically contain that information?
3. Which specific documents match those criteria?

Be thorough in your analysis. Consider document types, summaries, and key topics."""

    def _build_selection_prompt(self, question: str, reasoning: str, top_k: int) -> str:
        """
        Build prompt for selection step (uses tool to pick exact filenames).

        Args:
            question: User question
            reasoning: Reasoning from step 1
            top_k: Number of documents to select

        Returns:
            Formatted prompt string
        """
        prompt = f"""Based on your analysis:

{reasoning}

Now use the 'select_documents' tool to select the top {top_k} documents.

CRITICAL: You MUST use the select_documents tool and provide EXACT filenames from the available list.
Do not attempt to type filenames manually - use the tool to ensure exact matches."""

        return prompt

    def _get_selection_system_prompt(self) -> str:
        """Get system prompt for selection step."""
        return """You are finalizing document selection for a question answering system.

You have already analyzed which documents are relevant. Now you must use the select_documents tool to formally select them.

IMPORTANT: Always use the tool. The tool will validate that filenames are exact and valid."""

    def _get_selection_tools(self) -> List[Dict]:
        """
        Get tool definition for document selection.

        Returns:
            List containing tool definition
        """
        # Get all valid filenames
        valid_filenames = sorted(self.summaries.keys())

        tool = {
            "type": "function",
            "function": {
                "name": "select_documents",
                "description": "Select documents by their exact filenames. Only filenames from the provided list are valid.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "documents": {
                            "type": "array",
                            "description": "List of exact document filenames to select",
                            "items": {
                                "type": "string",
                                "enum": valid_filenames
                            }
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "Brief explanation of why these documents were selected"
                        }
                    },
                    "required": ["documents"]
                }
            }
        }

        return [tool]

    def _parse_tool_response(self, response: Dict, top_k: int) -> List[str]:
        """
        Parse tool calling response to extract selected documents.

        Args:
            response: Response from _call_llm_with_tools
            top_k: Expected number of documents

        Returns:
            List of document filenames
        """
        try:
            message = response.get("message", {})
            tool_calls = message.get("tool_calls", [])

            if not tool_calls:
                raise ValueError("No tool calls in response")

            # Get first tool call (should be select_documents)
            tool_call = tool_calls[0]
            function = tool_call.get("function", {})
            arguments = function.get("arguments", {})

            # Extract documents
            selected_docs = arguments.get("documents", [])

            if not selected_docs:
                raise ValueError("No documents in tool call arguments")

            # Take top_k
            selected_docs = selected_docs[:top_k]

            # Validate all documents exist (should be guaranteed by enum, but double-check)
            valid_docs = [doc for doc in selected_docs if doc in self.summaries]

            if not valid_docs:
                raise ValueError("No valid documents returned by tool")

            return valid_docs

        except Exception as e:
            raise ValueError(f"Failed to parse tool response: {e}")

    def get_document_summary(self, filename: str) -> Optional[Dict]:
        """
        Get summary for a specific document.

        Args:
            filename: Document filename

        Returns:
            Summary dictionary or None if not found
        """
        return self.summaries.get(filename)

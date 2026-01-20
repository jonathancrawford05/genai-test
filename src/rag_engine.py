"""
RAG query engine with LLM integration for answering questions.
"""
from typing import List, Optional
import os

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser

from .vector_store import VectorStore


class RAGEngine:
    """RAG engine that retrieves context and generates answers using LLM."""

    def __init__(
        self,
        vector_store: VectorStore,
        model_provider: str = "anthropic",
        model_name: str = "claude-3-5-sonnet-20241022",
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ):
        """
        Initialize RAG engine.

        Args:
            vector_store: Vector store for retrieval
            model_provider: "anthropic" or "openai"
            model_name: Model identifier
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
        """
        self.vector_store = vector_store
        self.model_provider = model_provider
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Initialize LLM
        self.llm = self._create_llm()

    def _create_llm(self):
        """Create LLM instance based on provider."""
        if self.model_provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not found in environment")

            return ChatAnthropic(
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                api_key=api_key,
            )

        elif self.model_provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment")

            return ChatOpenAI(
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                api_key=api_key,
            )

        else:
            raise ValueError(f"Unsupported model provider: {self.model_provider}")

    def answer_question(
        self,
        question: str,
        top_k: int = 5,
        include_sources: bool = True,
        filter_metadata: Optional[dict] = None,
    ) -> dict:
        """
        Answer a question using RAG.

        Args:
            question: User question
            top_k: Number of chunks to retrieve
            include_sources: Whether to include source citations
            filter_metadata: Optional metadata filter for retrieval

        Returns:
            Dictionary with answer and optional sources
        """
        # Retrieve relevant chunks
        results = self.vector_store.query(
            query_text=question,
            top_k=top_k,
            filter_metadata=filter_metadata,
        )

        if not results:
            return {
                "answer": "I couldn't find any relevant information to answer this question.",
                "sources": [],
                "num_sources": 0,
            }

        # Build context from retrieved chunks
        context_parts = []
        sources = []

        for i, result in enumerate(results, 1):
            doc = result["document"]
            metadata = result["metadata"]

            # Add to context
            context_parts.append(
                f"[Source {i}] {metadata['source_file']} (Page {metadata['page_number']}):\n{doc}"
            )

            # Track sources
            if include_sources:
                sources.append(
                    {
                        "source_file": metadata["source_file"],
                        "page_number": metadata["page_number"],
                        "relevance_score": 1 - result["distance"],  # Convert distance to similarity
                    }
                )

        context = "\n\n".join(context_parts)

        # Create prompt
        system_prompt = """You are a helpful AI assistant that answers questions based on provided context from PDF documents.

Guidelines:
- Answer the question using ONLY the information from the provided context
- Be precise and accurate
- If the context contains numerical data, extract it exactly as shown
- If the question asks for a list, format your response as a clear bulleted list
- If the question asks for a calculation, show your work
- If the context doesn't contain enough information, say so
- Cite sources by mentioning the document name and page number when relevant
- Do not make up information or use outside knowledge"""

        user_prompt = f"""Context from relevant documents:

{context}

Question: {question}

Answer:"""

        # Generate answer
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        response = self.llm.invoke(messages)
        answer = StrOutputParser().parse(response)

        return {
            "answer": answer.strip(),
            "sources": sources if include_sources else None,
            "num_sources": len(sources),
            "question": question,
        }

    def batch_answer(self, questions: List[str], **kwargs) -> List[dict]:
        """
        Answer multiple questions.

        Args:
            questions: List of questions
            **kwargs: Additional arguments for answer_question

        Returns:
            List of answer dictionaries
        """
        results = []
        for i, question in enumerate(questions, 1):
            print(f"Answering question {i}/{len(questions)}: {question[:60]}...")
            result = self.answer_question(question, **kwargs)
            results.append(result)

        return results

"""
Multi-agent RAG system for complex question answering.

Architecture:
- Router Agent: Selects relevant documents from summaries
- Planner Agent: Formulates multi-step retrieval strategy
- Retriever Agent: Executes retrieval from selected documents
"""

from .document_summarizer import DocumentSummarizer

__all__ = [
    "DocumentSummarizer",
]

"""
Multi-agent RAG system for complex question answering.

Architecture:
- Router Agent: Selects relevant documents from summaries
- Planner Agent: Formulates multi-step retrieval strategy
- Retriever Agent: Executes retrieval from selected documents
- Orchestrator Agent: Coordinates pipeline and generates answers
"""

from .document_summarizer import DocumentSummarizer
from .base_agent import BaseAgent, Message
from .router_agent import RouterAgent, RouterConfig
from .planner_agent import PlannerAgent, PlannerConfig, RetrievalPlan, RetrievalStep
from .retriever_agent import RetrieverAgent, RetrieverConfig, RetrievalResult, ExecutionResult
from .orchestrator_agent import OrchestratorAgent, OrchestratorConfig, Answer

__all__ = [
    "DocumentSummarizer",
    "BaseAgent",
    "Message",
    "RouterAgent",
    "RouterConfig",
    "PlannerAgent",
    "PlannerConfig",
    "RetrievalPlan",
    "RetrievalStep",
    "RetrieverAgent",
    "RetrieverConfig",
    "RetrievalResult",
    "ExecutionResult",
    "OrchestratorAgent",
    "OrchestratorConfig",
    "Answer",
]

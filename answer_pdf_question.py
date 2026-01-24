"""
Main interface for PDF question answering system.

This is the primary function for Part 1 of the GenAI test.
"""
from src.agents import OrchestratorAgent, OrchestratorConfig


def answer_pdf_question(question: str, pdfs_folder: str, verbose: bool = False) -> str:
    """
    Answer a question about PDF documents using multi-agent RAG system.

    This function implements the required interface for Part 1 of the test.
    It uses a multi-agent architecture:
    - Router: Selects relevant documents from summaries
    - Planner: Creates multi-step retrieval strategy
    - Retriever: Executes plan and retrieves relevant chunks
    - Orchestrator: Coordinates pipeline and generates final answer

    Args:
        question: A question about the content of the PDFs
        pdfs_folder: Path to a folder containing all the PDFs needed to answer the question
        verbose: Print intermediate steps and diagnostics (default: False)

    Returns:
        answer: Answer to the question as a string

    Example:
        >>> answer = answer_pdf_question(
        ...     question="What are the rules for an ineligible risk?",
        ...     pdfs_folder="artifacts/1"
        ... )
        >>> print(answer)
        The rules for an ineligible risk are...

    Note:
        Requires document_summaries.json in the pdfs_folder directory or
        artifacts/ directory. Run document summarization first if not present.
    """
    # Determine summaries path
    # Try pdfs_folder first, then artifacts/
    import os
    from pathlib import Path

    pdfs_path = Path(pdfs_folder)

    # Check for summaries in pdfs_folder
    summaries_in_folder = pdfs_path / "document_summaries.json"

    # Check for summaries in artifacts/
    summaries_in_artifacts = Path("artifacts") / "document_summaries.json"

    if summaries_in_folder.exists():
        summaries_path = str(summaries_in_folder)
    elif summaries_in_artifacts.exists():
        summaries_path = str(summaries_in_artifacts)
    else:
        raise FileNotFoundError(
            f"document_summaries.json not found in {pdfs_folder} or artifacts/.\n"
            f"Please run document summarization first:\n"
            f"  python -m src.agents.document_summarizer --pdf-folder {pdfs_folder}"
        )

    # Initialize orchestrator
    config = OrchestratorConfig(
        model="llama3.2",
        temperature=0.0,
        max_answer_tokens=2048
    )

    orchestrator = OrchestratorAgent(
        summaries_path=summaries_path,
        pdf_folder=pdfs_folder,
        config=config
    )

    # Get answer
    result = orchestrator.answer_question(question, verbose=verbose)

    # Return just the answer text (as required by spec)
    return result.answer


# Example usage
if __name__ == "__main__":
    # Example 1: EF_1 (ineligible risk rules)
    print("=" * 70)
    print("Example 1: Ineligible Risk Rules")
    print("=" * 70)

    question_1 = "What are the rules for an ineligible risk?"
    pdfs_folder_1 = "artifacts/1"

    answer_1 = answer_pdf_question(question_1, pdfs_folder_1, verbose=True)

    print("\n" + "=" * 70)
    print("FINAL ANSWER")
    print("=" * 70)
    print(answer_1)
    print()

    # Example 2: EF_2 (premium calculation)
    print("\n" + "=" * 70)
    print("Example 2: Premium Calculation")
    print("=" * 70)

    question_2 = "What is the premium for a Tier 1 building with Protection Class 5, Coverage A of $500,000 and 2% deductible?"
    pdfs_folder_2 = "artifacts/1"

    answer_2 = answer_pdf_question(question_2, pdfs_folder_2, verbose=True)

    print("\n" + "=" * 70)
    print("FINAL ANSWER")
    print("=" * 70)
    print(answer_2)
    print()

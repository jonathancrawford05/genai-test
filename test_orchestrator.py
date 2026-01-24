"""
Quick test of Orchestrator agent and answer_pdf_question interface.
"""
from answer_pdf_question import answer_pdf_question


def test_orchestrator():
    """Test the full pipeline with a simple question."""
    print("Testing Orchestrator with EF_1...")
    print()

    question = "What are the rules for an ineligible risk?"
    pdfs_folder = "artifacts/1"

    try:
        answer = answer_pdf_question(question, pdfs_folder, verbose=True)

        print("\n" + "=" * 70)
        print("SUCCESS - Answer Generated")
        print("=" * 70)
        print(f"Answer length: {len(answer)} characters")
        print(f"Preview: {answer[:500]}...")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_orchestrator()

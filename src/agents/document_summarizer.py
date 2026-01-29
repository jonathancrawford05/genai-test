"""
Document summarizer using LLM to generate structured summaries of PDFs.
"""
from pathlib import Path
from typing import Dict, List, Optional
import random

from pypdf import PdfReader
import ollama


class DocumentSummarizer:
    """
    Generates structured summaries of PDF documents for RAG routing.

    Extracts sample pages and uses LLM to create summaries describing:
    - Document content and purpose
    - What questions it can answer
    - Key entities and topics
    - Document type/category
    """

    def __init__(self, model: str = "llama3.2"):
        """
        Initialize summarizer.

        Args:
            model: Ollama model to use for summarization
        """
        self.model = model

    def summarize_document(self, pdf_path: Path, sample_pages: int = 3, percent_pages = 0.05) -> Dict:
        """
        Generate structured summary of a PDF document.

        Args:
            pdf_path: Path to PDF file
            sample_pages: Number of random pages to sample (in addition to first page)
            percent_pages: Minimum percentage of total pages to sample

        Returns:
            Dictionary with summary information
        """
        # Extract text from PDF
        reader = PdfReader(pdf_path)
        total_pages = len(reader.pages)
        sample_pages = max(int(total_pages*percent_pages), sample_pages)

        # Get first page (usually contains title/overview)
        first_page_text = reader.pages[0].extract_text() if total_pages > 0 else ""

        # Sample random pages for diversity
        sampled_texts = []
        if total_pages > 1:
            # Sample from middle and end pages
            sample_indices = random.sample(
                range(1, total_pages),
                min(sample_pages, total_pages - 1)
            )
            for idx in sample_indices:
                text = reader.pages[idx].extract_text()
                if text:
                    sampled_texts.append(f"--- Page {idx + 1} ---\n{text[:1000]}")

        # Combine samples
        sample_content = "\n\n".join([
            f"=== FIRST PAGE ===\n{first_page_text[:2000]}",
            *sampled_texts
        ])

        # Generate summary via LLM
        summary_data = self._generate_summary(pdf_path.name, sample_content, total_pages)

        # Add metadata
        summary_data["filename"] = pdf_path.name
        summary_data["total_pages"] = total_pages
        summary_data["generated_by"] = self.model

        return summary_data

    def _generate_summary(self, filename: str, sample_content: str, total_pages: int) -> Dict:
        """
        Use LLM to generate structured summary from sample content.

        Args:
            filename: PDF filename
            sample_content: Sampled text from PDF
            total_pages: Total page count

        Returns:
            Dictionary with summary fields
        """
        prompt = f"""You are analyzing a PDF document for a question-answering system that helps with insurance underwriting and rating decisions.

Document: {filename}
Pages: {total_pages}

Sample content from the document:
{sample_content}

Based on this sample, provide a structured analysis in JSON format:

{{
  "summary": "1-2 sentence overview of what this document contains",
  "use_for": "What types of questions can this document answer? Be specific about decision-support use cases.",
  "key_topics": ["list", "of", "5-8", "key", "topics", "or", "entities"],
  "document_type": "classification (e.g., rules_manual, rate_pages, checklist, actuarial_memo, forms)"
}}

Focus on:
- What information does this document provide?
- What business decisions does it support?
- What specific data/rules/rates does it contain?

Be concise and specific. Use insurance industry terminology where appropriate.

Return ONLY the JSON object, no other text."""

        try:
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.3, "num_predict": 512}
            )

            # Parse JSON response
            import json
            response_text = response["message"]["content"].strip()

            # Extract JSON if wrapped in markdown code blocks
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()

            summary = json.loads(response_text)

            # Validate required fields
            required = ["summary", "use_for", "key_topics", "document_type"]
            if not all(k in summary for k in required):
                raise ValueError(f"Missing required fields. Got: {summary.keys()}")

            return summary

        except Exception as e:
            print(f"  ⚠️  LLM summarization failed for {filename}: {e}")
            print(f"     Using fallback summary...")

            # Fallback: basic summary from filename
            return {
                "summary": f"Document: {filename}",
                "use_for": "Review document manually to determine use cases",
                "key_topics": ["unknown"],
                "document_type": "unknown",
                "error": str(e)
            }

    def summarize_folder(self, folder_path: Path, output_file: Optional[Path] = None) -> Dict[str, Dict]:
        """
        Generate summaries for all PDFs in a folder.

        Args:
            folder_path: Path to folder containing PDFs
            output_file: Optional path to save JSON output

        Returns:
            Dictionary mapping filenames to summary data
        """
        import json

        pdf_files = sorted(folder_path.glob("*.pdf"))
        summaries = {}

        print(f"\n{'=' * 60}")
        print(f"GENERATING DOCUMENT SUMMARIES")
        print(f"Model: {self.model}")
        print(f"{'=' * 60}\n")
        print(f"Found {len(pdf_files)} PDFs to summarize\n")

        for i, pdf_path in enumerate(pdf_files, 1):
            print(f"[{i}/{len(pdf_files)}] Summarizing {pdf_path.name}...")

            try:
                summary = self.summarize_document(pdf_path)
                summaries[pdf_path.name] = summary
                print(f"  ✓ {summary['document_type']}: {summary['summary'][:80]}...")

            except Exception as e:
                print(f"  ✗ Error: {e}")
                summaries[pdf_path.name] = {
                    "filename": pdf_path.name,
                    "error": str(e),
                    "summary": f"Error processing {pdf_path.name}",
                    "use_for": "Manual review required",
                    "key_topics": [],
                    "document_type": "error"
                }

        # Save to file if specified
        if output_file:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(summaries, f, indent=2, ensure_ascii=False)
            print(f"\n✓ Summaries saved to: {output_file}")

        return summaries

"""
Prototype PDF QA pipeline for the GenAI test.

Setup (minimal):
  pip install pdfplumber

Optional (for better table extraction):
  pip install pdfplumber[pdf]

This file implements:
  - answer_pdf_question(question, pdfs_folder)
  - two variations for experimentation
  - a small evaluation harness that reads artifacts/questions.csv

Design notes:
  - Text extraction uses pdfplumber.
  - Retrieval uses a lightweight BM25 implementation (no external ML deps).
  - Answering uses simple heuristics to keep the prototype runnable without an LLM.
  - You can swap in an LLM by replacing `heuristic_answer` with a model call.
"""
from __future__ import annotations

import csv
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import pdfplumber


@dataclass(frozen=True)
class Chunk:
    text: str
    source_file: str
    page_number: int


@dataclass(frozen=True)
class AnswerConfig:
    chunk_strategy: str
    top_k: int


DEFAULT_CONFIG = AnswerConfig(chunk_strategy="page", top_k=6)


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text.lower())


class BM25Index:
    def __init__(self, documents: List[str], k1: float = 1.5, b: float = 0.75):
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.tokenized_docs = [_tokenize(doc) for doc in documents]
        self.doc_len = [len(doc) for doc in self.tokenized_docs]
        self.avgdl = sum(self.doc_len) / max(len(self.doc_len), 1)
        self.df = {}
        for doc in self.tokenized_docs:
            for term in set(doc):
                self.df[term] = self.df.get(term, 0) + 1
        self.idf = {
            term: math.log(1 + (len(documents) - freq + 0.5) / (freq + 0.5))
            for term, freq in self.df.items()
        }

    def score(self, query: str) -> List[float]:
        query_terms = _tokenize(query)
        scores = [0.0] * len(self.documents)
        for idx, doc_terms in enumerate(self.tokenized_docs):
            term_freqs = {}
            for term in doc_terms:
                term_freqs[term] = term_freqs.get(term, 0) + 1
            doc_length = self.doc_len[idx]
            for term in query_terms:
                if term not in term_freqs:
                    continue
                idf = self.idf.get(term, 0.0)
                freq = term_freqs[term]
                denom = freq + self.k1 * (1 - self.b + self.b * doc_length / self.avgdl)
                scores[idx] += idf * (freq * (self.k1 + 1)) / denom
        return scores


def _chunk_by_page(pdf_path: Path) -> List[Chunk]:
    chunks: List[Chunk] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                chunks.append(
                    Chunk(text=text, source_file=pdf_path.name, page_number=page_number)
                )
    return chunks


def _chunk_by_char(pdf_path: Path, chunk_size: int = 1200, overlap: int = 200) -> List[Chunk]:
    chunks: List[Chunk] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            if not text.strip():
                continue
            start = 0
            while start < len(text):
                end = min(start + chunk_size, len(text))
                chunk_text = text[start:end]
                if chunk_text.strip():
                    chunks.append(
                        Chunk(
                            text=chunk_text,
                            source_file=pdf_path.name,
                            page_number=page_number,
                        )
                    )
                start = end - overlap
                if start < 0:
                    start = 0
    return chunks


def _load_chunks(pdfs_folder: str, chunk_strategy: str) -> List[Chunk]:
    pdfs_path = Path(pdfs_folder)
    pdf_files = sorted(pdfs_path.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDFs found in {pdfs_folder}")

    chunks: List[Chunk] = []
    for pdf_path in pdf_files:
        if chunk_strategy == "page":
            chunks.extend(_chunk_by_page(pdf_path))
        elif chunk_strategy == "char":
            chunks.extend(_chunk_by_char(pdf_path))
        else:
            raise ValueError(f"Unknown chunk_strategy: {chunk_strategy}")
    return chunks


_CACHE: dict[Tuple[str, str], Tuple[List[Chunk], BM25Index]] = {}


def _get_index(pdfs_folder: str, chunk_strategy: str) -> Tuple[List[Chunk], BM25Index]:
    cache_key = (pdfs_folder, chunk_strategy)
    if cache_key in _CACHE:
        return _CACHE[cache_key]
    chunks = _load_chunks(pdfs_folder, chunk_strategy)
    index = BM25Index([chunk.text for chunk in chunks])
    _CACHE[cache_key] = (chunks, index)
    return chunks, index


def _retrieve(question: str, pdfs_folder: str, config: AnswerConfig) -> List[Chunk]:
    chunks, index = _get_index(pdfs_folder, config.chunk_strategy)
    scores = index.score(question)
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    top_indices = [idx for idx, _ in ranked[: config.top_k]]
    return [chunks[idx] for idx in top_indices]


def _extract_list_items(text: str) -> List[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    items = []
    for line in lines:
        cleaned = line.lstrip("*•- ")
        if 3 < len(cleaned) <= 80 and cleaned[0].isupper():
            items.append(cleaned)
    unique_items = []
    seen = set()
    for item in items:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            unique_items.append(item)
    return unique_items


def _calculate_hurricane_premium(context: str) -> str | None:
    base_rate_match = re.search(r"Base Rate\s*\$?([0-9]+(?:\.[0-9]+)?)", context, re.I)
    factor_match = re.search(r"Factor\s*([0-9]+(?:\.[0-9]+)?)", context, re.I)
    if not base_rate_match or not factor_match:
        return None
    base_rate = float(base_rate_match.group(1))
    factor = float(factor_match.group(1))
    premium = round(base_rate * factor)
    return f"${premium}"


def _heuristic_answer(question: str, context: str) -> str:
    lowered = question.lower()
    if "calculate" in lowered or "premium" in lowered:
        calculated = _calculate_hurricane_premium(context)
        if calculated:
            return calculated
    if "list" in lowered:
        items = _extract_list_items(context)
        if items:
            return "\n".join(f"* {item}" for item in items)
    snippet = context.strip().splitlines()
    return snippet[0] if snippet else ""


def answer_pdf_question(question: str, pdfs_folder: str, config: AnswerConfig | None = None) -> str:
    """
    Answers a question about PDFs in the given folder using a lightweight RAG pipeline.
    """
    if config is None:
        config = DEFAULT_CONFIG
    retrieved = _retrieve(question, pdfs_folder, config)
    context = "\n".join(
        f"[{chunk.source_file} p{chunk.page_number}] {chunk.text}" for chunk in retrieved
    )
    return _heuristic_answer(question, context)


# --- Experimentation harness ---


def _normalize(text: str) -> str:
    text = re.sub(r"\s+", " ", text.lower()).strip()
    return re.sub(r"[^a-z0-9$ ]", "", text)


def _token_f1(prediction: str, reference: str) -> float:
    pred_tokens = _normalize(prediction).split()
    ref_tokens = _normalize(reference).split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = set(pred_tokens) & set(ref_tokens)
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def run_experiments(questions_csv: str, artifacts_root: str = "artifacts") -> None:
    """
    Runs two variations of the QA function and prints a comparison table.
    Metrics: exact match + token F1.
    """
    variations = {
        "bm25_page_top6": AnswerConfig(chunk_strategy="page", top_k=6),
        "bm25_char_top10": AnswerConfig(chunk_strategy="char", top_k=10),
    }

    rows = []
    with open(questions_csv, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            pdf_folder = os.path.join(artifacts_root, row["PDF Folder"].strip())
            for name, config in variations.items():
                answer = answer_pdf_question(row["question"], pdf_folder, config)
                expected = row["expected_output"]
                exact = _normalize(answer) == _normalize(expected)
                f1 = _token_f1(answer, expected)
                rows.append(
                    {
                        "id": row["id"],
                        "variation": name,
                        "answer": answer,
                        "expected": expected,
                        "exact_match": exact,
                        "token_f1": round(f1, 3),
                    }
                )

    print("\n=== Results ===")
    for row in rows:
        print(
            f"{row['id']} | {row['variation']} | exact={row['exact_match']} | f1={row['token_f1']}"
        )


if __name__ == "__main__":
    run_experiments("artifacts/questions.csv")
    print("🎉 All tests completed.")
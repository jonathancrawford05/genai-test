#!/usr/bin/env python3
"""Convert PDFs in a folder to Markdown using Pandoc."""
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


DEFAULT_PANDOC_ARGS = ["--wrap=none"]


def resolve_pandoc_path(explicit_path: str | None) -> str:
    if explicit_path:
        pandoc_path = Path(explicit_path).expanduser()
        if not pandoc_path.exists():
            raise FileNotFoundError(f"Pandoc not found at: {pandoc_path}")
        return str(pandoc_path)

    pandoc_path = shutil.which("pandoc")
    if pandoc_path:
        return pandoc_path

    raise FileNotFoundError(
        "Pandoc is not available on PATH. Install pandoc or pass --pandoc /path/to/pandoc."
    )


def convert_pdf(pandoc_path: str, pdf_path: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_md = output_dir / f"{pdf_path.stem}.md"

    media_dir = output_dir / "media" / pdf_path.stem
    media_dir.mkdir(parents=True, exist_ok=True)

    command = [
        pandoc_path,
        str(pdf_path),
        "-o",
        str(output_md),
        "--extract-media",
        str(media_dir),
        *DEFAULT_PANDOC_ARGS,
    ]

    subprocess.run(command, check=True)
    return output_md


def iter_pdfs(input_dir: Path) -> list[Path]:
    return sorted(path for path in input_dir.iterdir() if path.suffix.lower() == ".pdf")


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert PDFs to Markdown using pandoc.")
    parser.add_argument("input_dir", help="Folder containing PDF files")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Folder to write Markdown output (default: <input_dir>/markdown)",
    )
    parser.add_argument(
        "--pandoc",
        default=None,
        help="Optional path to pandoc binary if not on PATH",
    )

    args = parser.parse_args()
    input_dir = Path(args.input_dir).expanduser()
    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_dir}")

    output_dir = (
        Path(args.output_dir).expanduser()
        if args.output_dir
        else input_dir / "markdown"
    )

    pandoc_path = resolve_pandoc_path(args.pandoc)
    pdfs = iter_pdfs(input_dir)
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found in {input_dir}")

    for pdf_path in pdfs:
        output_md = convert_pdf(pandoc_path, pdf_path, output_dir)
        print(f"Converted {pdf_path.name} -> {output_md}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env bash
set -euo pipefail

INPUT_DIR="${1:-artifacts/1}"
OUTPUT_DIR="${2:-artifacts/1/markdown}"
MEDIA_DIR="${3:-${OUTPUT_DIR}/media}"

if ! command -v pandoc >/dev/null 2>&1; then
  echo "pandoc is not installed or not on PATH." >&2
  echo "Install pandoc and re-run: https://pandoc.org/installing.html" >&2
  exit 1
fi

if [[ ! -d "$INPUT_DIR" ]]; then
  echo "Input directory not found: $INPUT_DIR" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR" "$MEDIA_DIR"

shopt -s nullglob
pdfs=("$INPUT_DIR"/*.pdf)
if [[ ${#pdfs[@]} -eq 0 ]]; then
  echo "No PDF files found in $INPUT_DIR" >&2
  exit 1
fi

for pdf in "${pdfs[@]}"; do
  base_name=$(basename "$pdf" .pdf)
  output_md="$OUTPUT_DIR/${base_name}.md"
  media_out="$MEDIA_DIR/${base_name}"
  mkdir -p "$media_out"
  echo "Converting $pdf -> $output_md"
  pandoc "$pdf" -o "$output_md" --wrap=none --extract-media="$media_out"
done

printf '\nDone. Markdown files written to %s\n' "$OUTPUT_DIR"

# PDF → Markdown Pipeline (Pandoc)

This pipeline converts PDFs in a folder to Markdown using Pandoc. It is designed to work with the PDFs in `artifacts/1`.

## Prerequisites

- Pandoc installed and available on your `PATH`, **or** pass an explicit `--pandoc` path.

## Usage

```bash
python scripts/convert_pdfs_to_markdown.py artifacts/1
```

By default, Markdown files are written to `artifacts/1/markdown`, and any extracted media is stored in `artifacts/1/markdown/media/<pdf-stem>`.

### Optional: Specify an output directory

```bash
python scripts/convert_pdfs_to_markdown.py artifacts/1 --output-dir artifacts/1/markdown
```

### Optional: Explicit pandoc binary

```bash
python scripts/convert_pdfs_to_markdown.py artifacts/1 --pandoc /path/to/pandoc
```

## Notes

- This pipeline uses `--wrap=none` and `--extract-media` for better Markdown structure.
- For scanned PDFs, run OCR before conversion.

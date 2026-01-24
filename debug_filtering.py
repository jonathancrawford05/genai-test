"""Debug script to understand why filtering is failing."""

import chromadb
from src.onnx_processor import ONNXProcessor

# Initialize processor
processor = ONNXProcessor(
    persist_directory="./chroma_db_onnx",
    collection_name="pdf_documents"
)

# Query for content related to "Protection Class"
query = "Protection class factor for Tier 1 building with Protection Class 5"
results = processor.query(query, top_k=10)

print("=" * 70)
print("DEBUGGING FILTER ISSUE")
print("=" * 70)

print(f"\nQuery: {query}")
print(f"Retrieved: {len(results['documents'][0])} chunks")

# Target document from plan
target_doc = "(214933336-180358134)-Exhibit I - Non-Modeled Factors.pdf"
print(f"\nTarget document: {target_doc}")
print(f"Target document type: {type(target_doc)}")
print(f"Target as list: {[target_doc]}")

# Check what we actually got
print("\n" + "─" * 70)
print("Retrieved chunks and their source files:")
print("─" * 70)

for i, (doc_text, metadata) in enumerate(zip(
    results['documents'][0][:5],  # First 5
    results['metadatas'][0][:5]
)):
    source_file = metadata.get('source_file', '')
    page = metadata.get('page_number', '?')

    print(f"\n[{i+1}] Source: {source_file}")
    print(f"    Page: {page}")
    print(f"    Text: {doc_text[:100]}...")

    # Test filtering logic
    print(f"    Exact match test:")
    print(f"      source_file == target_doc: {source_file == target_doc}")
    print(f"      source_file in [target_doc]: {source_file in [target_doc]}")

    # Test partial matching
    clean_source = source_file.split(')-', 1)[-1] if ')-' in source_file else source_file
    clean_target = target_doc.split(')-', 1)[-1] if ')-' in target_doc else target_doc

    print(f"    Partial match test:")
    print(f"      clean_source: {clean_source}")
    print(f"      clean_target: {clean_target}")
    print(f"      clean_target in clean_source: {clean_target in clean_source}")
    print(f"      clean_source in clean_target: {clean_source in clean_target}")

print("\n" + "=" * 70)

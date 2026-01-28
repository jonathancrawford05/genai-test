# EF_2 Component-Based Scoring Framework

## Overview

This framework provides deterministic, component-based evaluation for the EF_2 question (Hurricane Premium Calculation) without requiring an LLM judge.

## Scoring Components

### Total: 100 Points

1. **Document Retrieval (20 points)**
   - 10 pts: Retrieved CT MAPS Homeowner Rules Manual
   - 10 pts: Retrieved CT Homeowners MAPS Rate Pages

2. **Deductible Identification (20 points)**
   - 15 pts: Identifies 2% deductible requirement
   - 5 pts: References Rule C-7 or >2,500 feet criterion

3. **Base Rate Identification (20 points)**
   - 15 pts: Finds $293 hurricane base rate
   - 5 pts: References Exhibit 1 or Page 4

4. **Factor Identification (20 points)**
   - 15 pts: Finds factor 2.061
   - 5 pts: References Exhibit 6, Page 71, or HO3/$750k/2% criteria

5. **Calculation (20 points)**
   - 20 pts: Correct final answer ($604 or ~603.87)
   - 10 pts: Shows correct formula even if inputs wrong

## Usage

### Basic Usage

```bash
./score_ef2_components.py /path/to/detailed_results_20260127_132344.json
```

This will:
- Analyze all EF_2 results in the JSON file
- Generate `ef2_component_scores.csv` with detailed scoring
- Print top 10 configurations and component statistics

### Custom Output Path

```bash
./score_ef2_components.py results.json --output my_scores.csv
```

### Show Top N Configurations

```bash
./score_ef2_components.py results.json --top 20
```

## Output Format

### CSV Columns

**Configuration:**
- `variation_name`: Experiment variation identifier
- `chunk_size`: Chunk size used (e.g., 1000, 2000)
- `top_k`: Top K chunks per retrieval step
- `expand_context`: Context window expansion
- `top_k_docs`: Number of documents selected by router
- `chunking_strategy`: Strategy (page, document, sliding_window)
- `use_hybrid`: Whether hybrid search was used
- `hybrid_alpha`: Hybrid search alpha parameter

**Component Scores (0-20 each):**
- `doc_retrieval`: Document retrieval score
- `deductible_id`: Deductible identification score
- `base_rate_id`: Base rate identification score
- `factor_id`: Factor identification score
- `calculation`: Calculation score
- `total_score`: Sum of all components (0-100)

**Reasoning:**
- `doc_retrieval_reason`: Why points were awarded/deducted
- `deductible_reason`: Deductible scoring explanation
- `base_rate_reason`: Base rate scoring explanation
- `factor_reason`: Factor scoring explanation
- `calculation_reason`: Calculation scoring explanation

**Metadata:**
- `answer_preview`: First 200 chars of answer
- `execution_time`: Time taken to generate answer
- `chunks_retrieved`: Total chunks retrieved

### Example Output

```
TOP 10 CONFIGURATIONS FOR EF_2 (by Component Score)
================================================================================
variation_name              chunk_size  top_k  total_score  doc_retrieval  deductible_id
chunk_2000_topk_5_docs1           2000      5         85.0           20.0           20.0
chunk_2000_topk_10_docs1          2000     10         80.0           20.0           15.0
chunk_1000_topk_5_docs2           1000      5         75.0           20.0           20.0
...
```

## Python API

You can also use the scorer programmatically:

```python
from score_ef2_components import EF2Scorer, analyze_results
import pandas as pd

# Analyze results
df = analyze_results("detailed_results.json", output_csv="scores.csv")

# Filter for specific configuration
high_scores = df[df['total_score'] >= 70]

# Score individual result
scorer = EF2Scorer()
result = {...}  # DetailedQuestionResult dict
scores, reasoning = scorer.score_result(result)

print(f"Total: {scores.total}/100")
print(f"Document Retrieval: {reasoning['document_retrieval']}")
```

## Interpreting Scores

### Score Ranges

- **90-100**: Excellent - Retrieved correct docs, identified all components, correct answer
- **70-89**: Good - Most components identified, may be missing references or final calculation
- **50-69**: Fair - Retrieved correct docs, partial component identification
- **30-49**: Poor - Missing key documents or most components
- **0-29**: Failed - Major retrieval or identification failures

### Common Patterns

**High Document Retrieval, Low Calculation:**
- System retrieved right documents but failed to extract correct values
- Suggests chunk retrieval or answer generation issues

**High Component ID, Low Document Retrieval:**
- Unlikely (can't identify components without documents)
- May indicate router selected wrong docs but retriever found right chunks anyway

**All Components Identified, Wrong Calculation:**
- System found all pieces but failed to multiply correctly
- Suggests answer generation or formatting issues

## Ground Truth Reference

Based on `artifacts/README.md`:

1. **Mandatory Hurricane Deductible**: 2%
   - Source: CT MAPS Homeowner Rules Manual, Rule C-7, Page 23
   - Criteria: Coastline Neighborhood, >2,500 feet from coast

2. **Hurricane Base Rate**: $293
   - Source: CT Homeowners MAPS Rate Pages, Page 4, Exhibit 1

3. **Deductible Factor**: 2.061
   - Source: CT Homeowners MAPS Rate Pages, Page 71, Exhibit 6
   - Criteria: HO3, Coverage A = $750,000, Deductible = 2%

4. **Calculation**: 293 × 2.061 = 603.873 ≈ **$604**

## Validation

The scorer uses keyword matching for deterministic evaluation:

- **2% Deductible**: Searches for "2%", "2 percent", etc.
- **$293 Base Rate**: Searches for "293", "$293"
- **Factor 2.061**: Searches for "2.061", "2.06"
- **Final Answer**: Searches for "604", "$604", "603.87"

This approach is:
- ✓ Fast and deterministic
- ✓ No API costs (unlike LLM judge)
- ✓ Reproducible
- ✓ Component-level insights
- ⚠ May miss paraphrased or formatted numbers differently
- ⚠ Requires exact number matches

## Future Enhancements

If keyword matching proves insufficient, consider:

1. **Fuzzy number matching**: Regex for "$X" or "X dollars"
2. **Calculation validation**: Parse numbers and verify math
3. **LLM judge for edge cases**: Hybrid approach - keywords for most, LLM for ambiguous
4. **Page number validation**: Check if correct pages mentioned in retrieval steps

## Examples

### Perfect Score (100/100)

```
Answer: "Based on Rule C-7, the property is more than 2,500 feet from the
coastline, requiring a mandatory 2% hurricane deductible. From Exhibit 1
(Page 4), the hurricane base rate is $293. Exhibit 6 (Page 71) shows the
factor for HO3 with $750,000 coverage and 2% deductible is 2.061.
The premium is: $293 × 2.061 = $604"

Components:
- Document Retrieval: 20/20 (both docs retrieved)
- Deductible ID: 20/20 (2%, Rule C-7, distance)
- Base Rate ID: 20/20 ($293, Exhibit 1)
- Factor ID: 20/20 (2.061, Exhibit 6, criteria)
- Calculation: 20/20 (correct answer)
```

### Partial Score (65/100)

```
Answer: "The hurricane deductible is 2% based on the distance from the coast.
The base rate is $293. The factor should be applied to calculate the premium."

Components:
- Document Retrieval: 20/20 (both docs retrieved)
- Deductible ID: 15/20 (2% identified, no Rule C-7 reference)
- Base Rate ID: 15/20 ($293 found, no Exhibit reference)
- Factor ID: 0/20 (factor not mentioned)
- Calculation: 0/20 (no final answer)
```

### Low Score (25/100)

```
Answer: "The hurricane premium depends on various factors including
deductible and base rates from the policy documents."

Components:
- Document Retrieval: 20/20 (correct docs selected)
- Deductible ID: 0/20 (no specific deductible mentioned)
- Base Rate ID: 0/20 (no specific rate)
- Factor ID: 0/20 (no factor)
- Calculation: 0/20 (no calculation)
```

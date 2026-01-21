# RAG Query Guide

Multiple ways to query your indexed PDFs, from free local models to premium APIs.

## Quick Start

**Fastest (Free, Local):**
```bash
python query_ollama.py "What are the rating plan rules?"
```

## All Query Options

### 1. Ollama (Recommended - Free & Local)

**Default (llama3.2):**
```bash
python query_ollama.py "List all rating plan rules"
```

**Specify model:**
```bash
python query_ollama.py --model mistral "What is the base rate?"
```

**Popular Ollama models:**
- `llama3.2` - Fast, good quality (default)
- `llama3.2:3b` - Smaller, faster
- `mistral` - Fast and accurate
- `gemma2` - Google's model
- `qwen2.5` - Strong reasoning

**Install models:**
```bash
ollama pull llama3.2
ollama pull mistral
ollama list  # See installed models
```

### 2. Unified Script (All Providers)

**Ollama:**
```bash
python query_rag.py "your question"
python query_rag.py --provider ollama --model llama3.2 "your question"
```

**OpenAI:**
```bash
python query_rag.py --provider openai --model gpt-4o-mini "your question"
```

**Anthropic:**
```bash
python query_rag.py --provider anthropic --model claude-3-5-sonnet-20241022 "your question"
```

### 3. Keyword Search (No LLM, Instant)

**Just retrieval, no AI answer:**
```bash
python query_light.py "your question"
```

Returns top 5 relevant chunks without generating an answer.

### 4. OpenAI Only

**If you only want OpenAI:**
```bash
export OPENAI_API_KEY="sk-..."
python query_openai.py "your question"
```

## Comparison

| Method | Speed | Cost | Quality | Requirements |
|--------|-------|------|---------|--------------|
| **Ollama** | Medium | Free | Good | Ollama installed |
| **Keyword** | Instant | Free | N/A | None |
| **OpenAI** | Fast | ~$0.01/query | Excellent | API key + credits |
| **Anthropic** | Fast | ~$0.03/query | Excellent | API key + credits |

## Setup Requirements

### For Ollama (Recommended)

1. **Install Ollama:**
   ```bash
   # macOS
   brew install ollama

   # Or download from https://ollama.ai
   ```

2. **Start Ollama server:**
   ```bash
   ollama serve
   ```

3. **Pull a model:**
   ```bash
   ollama pull llama3.2
   ```

4. **Query:**
   ```bash
   python query_ollama.py "your question"
   ```

### For OpenAI

1. **Add API key to .env:**
   ```bash
   OPENAI_API_KEY=sk-...
   ```

2. **Add credits** to your OpenAI account

3. **Query:**
   ```bash
   python query_rag.py --provider openai "your question"
   ```

### For Anthropic

1. **Add API key to .env:**
   ```bash
   ANTHROPIC_API_KEY=sk-ant-...
   ```

2. **Add credits** to your Anthropic account

3. **Query:**
   ```bash
   python query_rag.py --provider anthropic "your question"
   ```

## Testing with Questions CSV

**Test all questions from your CSV file:**

**With Ollama (free):**
```bash
# Coming soon - batch processing script
```

**With OpenAI:**
```bash
python test_questions.py
```

## Troubleshooting

### Ollama Issues

**"Connection refused":**
```bash
# Start Ollama server
ollama serve
```

**"Model not found":**
```bash
# Pull the model first
ollama pull llama3.2
```

**"Out of memory":**
```bash
# Use a smaller model
python query_ollama.py --model llama3.2:3b "your question"
```

### OpenAI Issues

**"Error code: 429 - insufficient_quota":**
- Add credits to your OpenAI account
- Or use Ollama (free)

**"OPENAI_API_KEY not found":**
- Add to .env file: `OPENAI_API_KEY=sk-...`

### Anthropic Issues

**"Error: API key not found":**
- Add to .env file: `ANTHROPIC_API_KEY=sk-ant-...`

## Examples

### Question 1: List all rating plan rules

```bash
python query_ollama.py "List all rating plan rules"
```

**Output:**
```
============================================================
ANSWER
============================================================
Based on the provided documents, here are all the rating plan rules:

* Limits of Liability and Coverage Relationships
* Rating Perils
* Base Rates
* Policy Type Factor
...

============================================================
SOURCES
============================================================
1. CT MAPS Homeowner Rules Manual.pdf (page 12) [relevance: 0.892]
2. Checklist - HO Rules.pdf (page 3) [relevance: 0.845]
...
```

### Question 2: Calculate premium

```bash
python query_ollama.py "Using the Base Rate and the applicable Mandatory Hurricane Deductible Factor, calculate the unadjusted Hurricane premium for an HO3 policy with a \$750,000 Coverage A limit located 3,000 feet from the coast in a Coastline Neighborhood."
```

**Output:**
```
ANSWER: $604
SOURCES: CT Homeowners MAPS Rate Pages.pdf (page 45)
```

## Performance

| Operation | Time | Memory | Cost |
|-----------|------|--------|------|
| Index 22 PDFs | 62 sec | 500MB | Free |
| Query (Ollama) | 2-5 sec | 500MB | Free |
| Query (OpenAI) | 2-3 sec | 500MB | ~$0.01 |
| Query (Anthropic) | 2-3 sec | 500MB | ~$0.03 |

## Recommendations

**For development/testing:**
- Use **Ollama** (free, no API limits)
- Model: `llama3.2` or `mistral`

**For production:**
- Use **OpenAI gpt-4o-mini** (cheap, fast, reliable)
- Or **Ollama** if you have a powerful server

**For best quality:**
- Use **Anthropic Claude 3.5 Sonnet**
- Or **OpenAI GPT-4**

**For instant results (no AI):**
- Use **query_light.py** (keyword search only)

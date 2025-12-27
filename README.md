# Mini RAG System for Construction Marketplace

**Assignment: Building a Retrieval-Augmented Generation Pipeline**

A complete RAG (Retrieval-Augmented Generation) chatbot implementation for Indecimal, a construction marketplace. This system retrieves relevant information from internal documents and generates precise, grounded answers using a local LLM.

---

## Table of Contents

1. [Overview](#overview)
2. [Problem Statement](#problem-statement)
3. [Solution Architecture](#solution-architecture)
4. [Technical Implementation](#technical-implementation)
5. [Design Decisions & Rationale](#design-decisions--rationale)
6. [Project Structure](#project-structure)
7. [Installation & Setup](#installation--setup)
8. [Usage Guide](#usage-guide)
9. [Evaluation & Quality Analysis](#evaluation--quality-analysis)
10. [Limitations & Future Improvements](#limitations--future-improvements)

---

## Overview

This project implements a **Mini RAG Pipeline** that enables users to ask natural language questions about Indecimal's construction services and receive accurate, source-cited answers derived exclusively from provided internal documents.

### Key Features

- **Document Processing**: Intelligent markdown-aware chunking that preserves semantic coherence
- **Semantic Search**: FAISS vector similarity search for retrieving relevant document chunks
- **Grounded Generation**: LLM responses strictly constrained to retrieved context
- **Transparency**: Clear display of source documents and relevance scores
- **Local Execution**: Runs entirely offline using Ollama for LLM inference

---

## Problem Statement

Build a simple Retrieval-Augmented Generation (RAG) pipeline for a construction marketplace AI assistant that:

1. Chunks and embeds provided internal documents
2. Implements semantic retrieval using a vector store
3. Uses an LLM to generate answers grounded in retrieved content
4. Demonstrates clarity, correctness, and explainability
5. Provides a working chatbot interface

### Input Documents

| Document | Content | Purpose |
|----------|---------|---------|
| `doc1.md` | Company overview, customer journey, FAQs | General information |
| `doc2.md` | Package pricing, material specifications | Technical details |
| `doc3.md` | Policies, quality assurance, maintenance | Trust & guarantees |

---

## Solution Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER QUERY                               │
│                    "What is the Premier package price?"          │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    EMBEDDING GENERATION                          │
│              Model: all-MiniLM-L6-v2 (384 dimensions)           │
│                    Query → Dense Vector                          │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    VECTOR SIMILARITY SEARCH                      │
│                    FAISS IndexFlatL2                             │
│              Find top-3 most similar document chunks             │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    CONTEXT CONSTRUCTION                          │
│              Combine retrieved chunks with metadata              │
│              (source file, section name, relevance score)        │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LLM GENERATION                                │
│              Model: llama3.2:1b via Ollama                       │
│              Strictly grounded prompt with citations             │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    RESPONSE                                      │
│    "The Premier package price is ₹1,995/sqft (incl. GST)"       │
│                    [Source: doc2.md - Package Pricing]           │
└─────────────────────────────────────────────────────────────────┘
```

---

## Technical Implementation

### 1. Document Chunking Strategy

**Approach**: Header-based chunking using markdown structure

```python
# Chunks are split at ## and ### headers
if line.startswith('## ') or line.startswith('### '):
    # Start new chunk with section context
    current_section = line.lstrip('#').strip()
```

**Rationale**: 
- Preserves semantic coherence within each chunk
- Maintains section context for better retrieval accuracy
- Avoids splitting related content across chunks

**Result**: 41 chunks generated from 3 documents

### 2. Embedding Model

**Model**: `sentence-transformers/all-MiniLM-L6-v2`

| Property | Value |
|----------|-------|
| Dimension | 384 |
| Model Size | ~80MB |
| Execution | Local (CPU) |
| Speed | Fast |

**Why this model?**
- Lightweight enough to run locally without GPU
- Good semantic understanding for English text
- No API costs or external dependencies
- Widely used and well-documented

### 3. Vector Store (FAISS)

**Index Type**: `IndexFlatL2` (Flat L2 distance)

```python
self.index = faiss.IndexFlatL2(embedding_dim)
self.index.add(embeddings)
```

**Characteristics**:
- Exact nearest neighbor search (no approximation)
- Suitable for small datasets (<10K vectors)
- No training required
- O(n) search complexity

**Similarity Scoring**:
```python
# Convert L2 distance to similarity score (0-1)
similarity = 1 / (1 + distance)
```

### 4. Retrieval Function

```python
def retrieve(self, query: str, top_k: int = 3):
    # 1. Embed the query
    query_embedding = self.embedding_model.encode([query])
    
    # 2. Search FAISS index
    distances, indices = self.index.search(query_embedding, top_k)
    
    # 3. Return chunks with similarity scores
    return [(chunk, similarity) for chunk, similarity in results]
```

**Default Parameters**:
- `top_k = 3`: Retrieve 3 most relevant chunks
- Provides sufficient context without overwhelming the LLM

### 5. LLM Integration (Ollama)

**Model**: `llama3.2:1b`

**Grounding Prompt**:
```
You are Indecimal's AI assistant. Answer questions ONLY using the 
provided context. If the information is not in the context, say 
"I don't have information about that in my knowledge base."

RULES:
1. Use ONLY information from the provided context
2. Cite sources using [Source N: filename - section]
3. Be concise and accurate
4. Never make up information
```

**Why local LLM?**
- No API costs
- Privacy (data never leaves the machine)
- Faster iteration during development
- Works offline

### 6. User Interface (Streamlit)

**Features**:
- Clean chat interface with message history
- Sidebar with system statistics
- Example questions for quick testing
- Retrieved context display with relevance scores
- Scrollable message container
- Expandable raw JSON response

---

## Design Decisions & Rationale

### 1. Why Header-Based Chunking?

| Alternative | Pros | Cons | Decision |
|-------------|------|------|----------|
| Fixed-size (500 chars) | Simple | Breaks mid-sentence | ❌ |
| Sentence-based | Natural breaks | Too granular | ❌ |
| Paragraph-based | Good coherence | Variable quality | ❌ |
| **Header-based** | **Semantic units** | **Requires markdown** | ✅ |

Since input documents are markdown with clear section headers, header-based chunking preserves the logical structure of the content.

### 2. Why all-MiniLM-L6-v2?

Compared alternatives:
- `text-embedding-ada-002` (OpenAI): Requires API, costs money
- `all-mpnet-base-v2`: Larger (420MB), slower
- **`all-MiniLM-L6-v2`**: Good balance of quality, size, and speed

### 3. Why FAISS over ChromaDB/Pinecone?

- **FAISS**: Local, no setup, fast, well-suited for small datasets
- ChromaDB: More features but heavier dependency
- Pinecone: Requires account/API, overkill for this use case

### 4. Why Ollama over OpenAI API?

- **Free**: No API costs
- **Private**: Data stays local
- **Reliable**: No rate limits or network issues
- **Controllable**: Can switch models easily

---

## Project Structure

```
Indecimel-RAG/
│
├── doc1.md                 # Company overview document
├── doc2.md                 # Package specifications document
├── doc3.md                 # Policies & guarantees document
│
├── rag_pipeline.py         # Core RAG implementation
│   ├── DocumentChunk       # Dataclass for chunk metadata
│   ├── RAGPipeline         # Main pipeline class
│   │   ├── _chunk_document()    # Header-based chunking
│   │   ├── _load_and_index()    # Build FAISS index
│   │   ├── retrieve()           # Vector similarity search
│   │   ├── generate_answer()    # LLM generation
│   │   └── query()              # End-to-end RAG
│   └── get_stats()         # System statistics
│
├── app.py                  # Streamlit chat interface
│   ├── render_header()     # Logo and title
│   ├── render_context_card() # Retrieved chunk display
│   └── main()              # Application entry point
│
├── evaluate.py             # Quality evaluation script
│   ├── TEST_QUESTIONS      # 15 test cases
│   ├── evaluate_retrieval() # Check source accuracy
│   ├── evaluate_answer()   # Check grounding
│   └── run_evaluation()    # Generate report
│
├── requirements.txt        # Python dependencies
├── indecimallogo.jpg       # Company logo
└── README.md               # This file
```

---

## Installation & Setup

### Prerequisites

- Python 3.9+
- Ollama (for local LLM)

### Step 1: Clone/Download Repository

```bash
cd D:\Indecimel-RAG
```

### Step 2: Install Python Dependencies

```bash
pip install -r requirements.txt
```

**Dependencies**:
- `sentence-transformers`: Embedding model
- `faiss-cpu`: Vector similarity search
- `streamlit`: Web UI framework
- `ollama`: Local LLM client

### Step 3: Install Ollama

1. Download from [ollama.com](https://ollama.com/)
2. Install and launch the application
3. Pull the required model:

```bash
ollama pull llama3.2:1b
```

*Note: The 1b model is ~1.3GB and requires minimal GPU memory*

### Step 4: Run the Application

```bash
python -m streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## Usage Guide

### Asking Questions

1. Type your question in the chat input at the bottom
2. Press Enter or click the send button
3. Wait for the response (typically 5-15 seconds)

### Example Queries

| Query | Expected Source |
|-------|-----------------|
| "What is the Premier package price?" | doc2.md |
| "How do you handle construction delays?" | doc3.md |
| "Explain the customer journey" | doc1.md |
| "What quality checks are performed?" | doc1.md, doc3.md |
| "What maintenance is included?" | doc3.md |

### Understanding the Response

- **Answer**: Generated text based on retrieved context
- **Source Citations**: [Source N: filename - section]
- **Retrieved Context**: Expandable cards showing the actual chunks used
- **Relevance Scores**: Percentage match for each chunk

---

## Evaluation & Quality Analysis

### Running Evaluation

```bash
python evaluate.py
```

### Test Suite

The evaluation includes 15 test questions covering:
- Package pricing queries
- Policy and guarantee questions
- Customer journey inquiries
- Quality assurance details
- Maintenance coverage

### Evaluation Metrics

1. **Retrieval Accuracy**: Did we retrieve the correct source document?
2. **Topic Coverage**: Does the answer address the expected topics?
3. **Grounding**: Are answers citing sources appropriately?
4. **Response Time**: Latency per query

### Sample Results

```
Question: What is the Premier package price?
Retrieved: doc2.md - Package Pricing (49% match)
Answer: The Premier package price is ₹1,995/sqft (incl. GST).
Status: ✓ Correct source, ✓ Accurate answer
```

### Evaluation Summary & Observations

Based on running the 15 test questions through the RAG pipeline:

| Metric | Result |
|--------|--------|
| Source Retrieval Accuracy | ~87% (correct document retrieved) |
| Topic Coverage (Retrieval) | ~65% (relevant topics in chunks) |
| Topic Coverage (Answer) | ~45% (topics mentioned in response) |
| Answers with Citations | ~80% (includes source references) |
| Average Response Time | ~8-12 seconds per query |

**Key Observations:**

1. **Strong Retrieval**: The FAISS index successfully retrieves the correct source document for most queries. Header-based chunking preserves semantic boundaries effectively.

2. **Grounding Works**: The LLM follows the grounding instructions well, citing sources in most responses and avoiding fabricated information. When context is insufficient, it appropriately indicates missing information.

3. **Topic Coverage Gap**: While retrieved chunks contain relevant content (~65% topic coverage), the final answers capture fewer topics (~45%). This is expected as the LLM summarizes rather than exhaustively listing all details.

4. **Latency Considerations**: Using `llama3.2:1b` locally provides reasonable response times (8-12s). Larger models would improve answer quality but increase latency.

5. **No Hallucinations Detected**: In testing, the model did not generate information that wasn't present in the retrieved context, demonstrating effective grounding.

**Potential Improvements:**
- Increase `top_k` from 3 to 5 for broader context
- Add chunk overlap to avoid missing boundary information
- Use a larger LLM (3B+) for more comprehensive answers

---

## Limitations & Future Improvements

### Current Limitations

1. **Small Dataset**: Only 3 documents, 41 chunks
2. **Single Language**: English only
3. **No Conversation Memory**: Each query is independent
4. **Simple Chunking**: Header-based may miss cross-section context
5. **No Fine-tuning**: Using generic embedding model

### Potential Improvements

1. **Hybrid Search**: Combine BM25 keyword search with semantic search
2. **Re-ranking**: Use a cross-encoder to re-rank retrieved chunks
3. **Chunking Overlap**: Add overlapping windows for better context
4. **Conversation History**: Track multi-turn conversations
5. **Caching**: Cache embeddings and frequent queries
6. **Better UI**: Add chat history persistence, export functionality

---

## Conclusion

This RAG implementation demonstrates the core principles of retrieval-augmented generation:

1. **Chunking**: Breaking documents into semantically meaningful units
2. **Embedding**: Converting text to dense vector representations
3. **Retrieval**: Finding relevant content via similarity search
4. **Grounding**: Constraining LLM output to retrieved context

The system successfully answers questions about Indecimal's construction services using only the provided documents, with clear transparency about the sources used.

---

## References

- [Sentence Transformers Documentation](https://www.sbert.net/)
- [FAISS Wiki](https://github.com/facebookresearch/faiss/wiki)
- [Ollama Documentation](https://ollama.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [RAG Paper (Lewis et al., 2020)](https://arxiv.org/abs/2005.11401)

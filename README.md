# Multi-Agent RAG for Systematic Literature Review

A **Multi-Agent RAG System** that orchestrates specialized AI agents to perform systematic literature reviews — ingesting PDFs, expanding queries, retrieving and screening papers, writing structured sections, and exporting polished reports.

## 🏗️ Architecture

```
Query
  │
  ▼
expand_query ──► retrieve (local FAISS + optional ArXiv)
  │                  │
  │                  ▼
  │            rerank (cross-encoder)
  │                  │
  │                  ▼
  │              screen (ScreeningCouncil: 2-of-3 vote)
  │                  │
  └──◄── retry ◄── supervisor_gate ──► write_methods
    (max 1)                                │
                                      write_results
                                           │
                                      write_challenges
                                           │
                                      merge_sections
                                           │
                                      critique_review
                                           │
                          revise ◄────────┤────────► format_output ──► END
                          (max 1)                          │
                                                         export
                                                        (MD/BibTeX/JSON)
```

### Agents

| Agent | File | Description |
|---|---|---|
| QueryExpansion | `agents/query_expansion.py` | Generates 3-5 diverse queries from one topic |
| ArxivSearch | `agents/web_search.py` | Live ArXiv paper discovery (opt-in) |
| ScreeningCouncil | `agents/screening_council.py` | 3 screeners (recency, empirical, methodology) — majority vote |
| Supervisor | `agents/supervisor.py` | Heuristic quality gate; triggers retry if too few docs |
| SectionWriters | `agents/section_writers.py` | Focused writers for Methods, Results, Challenges |
| Critic | `agents/critic.py` | Validates citations in the draft; triggers revision |
| Aggregator | `agents/aggregator.py` | Full-review synthesis (used in revision) |

### Key Features
- **Deduplication**: MD5 content hashing at ingestion
- **Re-ranking**: Cross-encoder (`ms-marco-MiniLM-L-6-v2`) before screening
- **LLM Caching**: All LLM calls cached on disk (`diskcache`) — re-runs are instant
- **Rich Metadata**: Title/authors/year from Semantic Scholar API, PDF text, or filename fallback
- **Structured Export**: Markdown, BibTeX, and JSON reports in `outputs/`
- **Evaluation**: Context precision, recall, faithfulness metrics printed after every run

## 🚀 Tech Stack

| Component | Technology |
|---|---|
| LLM | Ollama (local) — supports any model, default `llama2` |
| Orchestration | LangGraph |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Vector Store | FAISS (`faiss-cpu`) |
| PDF Parsing | PyMuPDF + Semantic Scholar API |
| Caching | `diskcache` |

## 📂 Project Structure

```
├── agents/
│   ├── query_expansion.py    # Expand query into sub-queries
│   ├── screening_council.py  # 3-screener voting council
│   ├── supervisor.py         # Quality gate (heuristic)
│   ├── section_writers.py    # Methods / Results / Challenges writers
│   ├── critic.py             # Citation validation
│   ├── aggregator.py         # Full synthesis (revision fallback)
│   └── web_search.py         # ArXiv live search
├── rag/
│   ├── ingest.py             # PDF loading, chunking, metadata extraction
│   ├── embed.py              # Sentence-transformer embeddings
│   ├── index.py              # FAISS vector store with deduplication
│   ├── retriever.py          # k-NN retrieval
│   └── reranker.py           # Cross-encoder re-ranking
├── orchestration/
│   └── graph.py              # LangGraph state machine
├── utils/
│   ├── cache.py              # Disk-based LLM response cache
│   ├── exporter.py           # Markdown / BibTeX / JSON export
│   └── evaluator.py          # RAG evaluation metrics
├── data/                     # PDFs + FAISS index
├── outputs/                  # Generated reports
└── run_demo.py               # CLI entry point
```

## ⚡ Quick Start

### 1. Prerequisites

Install [Ollama](https://ollama.com) and pull a model:
```bash
ollama pull llama2        # default (requires ~4GB RAM)
# or for lower memory:
ollama pull tinyllama     # ~660MB
```
Update `.env` if using a non-default model:
```
OLLAMA_MODEL=tinyllama
OLLAMA_BASE_URL=http://localhost:11434
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the System

```bash
# Basic run (ingest PDFs in data/ then query)
python run_demo.py --query "Machine unlearning in federated learning"

# Skip ingestion if already indexed
python run_demo.py --query "Local Differential Privacy" --skip-ingest

# Export results as Markdown, BibTeX and JSON
python run_demo.py --query "..." --skip-ingest --export

# Add live ArXiv paper search
python run_demo.py --query "..." --skip-ingest --arxiv

# Pause after screening for human review of papers
python run_demo.py --query "..." --skip-ingest --human-review

# Full run with all features
python run_demo.py \
  --query "Methods for Machine Unlearning in Federated Learning" \
  --arxiv --human-review --export
```

### CLI Flags

| Flag | Default | Purpose |
|---|---|---|
| `--query` | `"Methods for unlearning in FL"` | Research question |
| `--data-dir` | `data/` | Directory containing PDFs |
| `--persist-dir` | `data/faiss_index` | Vector store location |
| `--skip-ingest` | off | Skip PDF ingestion |
| `--export` | off | Save MD/BibTeX/JSON to `outputs/` |
| `--arxiv` | off | Fetch live ArXiv papers |
| `--human-review` | off | Pause after screening for manual review |
| `--threshold` | `3` | Min screened docs before synthesis |

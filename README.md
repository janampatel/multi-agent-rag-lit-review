# Multi-Agent RAG for Systematic Literature Review

A **Multi-Agent RAG Web Application** that orchestrates specialized AI agents to perform systematic literature reviews — ingesting PDFs, expanding queries, retrieving and screening papers, writing structured sections, and exporting polished reports.

## 🌐 Web Interface

**NEW:** Now available as a modern web application with:
- **FastAPI Backend** - RESTful API with async job processing
- **Next.js Frontend** - Interactive UI with real-time progress tracking
- **Live Progress Updates** - Watch AI agents work in real-time
- **Document Management** - Upload and manage PDFs via web interface

## 🏗️ Architecture

```
Web UI (Next.js)
     │
     ▼
FastAPI Backend
     │
     ▼
Query → expand_query → retrieve (FAISS + optional ArXiv)
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
                                                            add_references
                                                                 │
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
| **Backend** | FastAPI with async job processing |
| **Frontend** | Next.js 14 with TypeScript & Tailwind CSS |
| **LLM** | **Backboard API** (Cohere, OpenAI, Anthropic, Google) — cloud-hosted with persistent memory |
| **Orchestration** | LangGraph |
| **Embeddings** | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| **Vector Store** | FAISS (`faiss-cpu`) |
| **PDF Parsing** | PyMuPDF + Semantic Scholar API |
| **Memory** | Backboard persistent memory (cross-session) |
| **Caching** | Disk-based LLM response cache |

## 📂 Project Structure

```
├── backend/                   # FastAPI web server
│   └── main.py              # REST API endpoints
├── frontend/                  # Next.js web interface
│   ├── app/                 # Next.js pages
│   │   ├── page.tsx         # Main UI
│   │   ├── layout.tsx       # App layout
│   │   └── globals.css      # Styles & animations
│   ├── components/          # Reusable UI components
│   └── package.json         # Frontend dependencies
├── agents/
│   ├── query_expansion.py   # Expand query into sub-queries
│   ├── screening_council.py # 3-screener voting council
│   ├── supervisor.py        # Quality gate (heuristic)
│   ├── section_writers.py   # Methods / Results / Challenges writers
│   ├── critic.py            # Citation validation
│   ├── aggregator.py        # Full synthesis (revision fallback)
│   └── web_search.py        # ArXiv live search
├── rag/
│   ├── ingest.py            # PDF loading, chunking, metadata extraction
│   ├── embed.py             # Sentence-transformer embeddings
│   ├── index.py             # FAISS vector store with deduplication
│   ├── retriever.py         # k-NN retrieval with auto-path detection
│   └── reranker.py          # Cross-encoder re-ranking
├── orchestration/
│   └── graph.py             # LangGraph state machine
├── utils/
│   ├── cache.py             # Disk-based LLM response cache
│   ├── backboard_client.py  # Backboard API client
│   ├── backboard_langchain.py # LangChain wrapper
│   ├── exporter.py          # Markdown / BibTeX / JSON export
│   └── evaluator.py         # RAG evaluation metrics
├── data/                      # PDFs + FAISS index
├── outputs/                   # Generated reports
├── start_webapp.bat           # One-click startup script
└── run_demo.py                # CLI entry point (legacy)
```

## ⚡ Quick Start

### 1. Prerequisites

**Get Backboard API Key:**
1. Sign up at [backboard.io](https://backboard.io)
2. Navigate to Settings → API Keys
3. Create and copy your API key

Update `.env`:
```env
BACKBOARD_API_KEY=your_api_key_here
BACKBOARD_MODEL_PROVIDER=cohere
BACKBOARD_MODEL_NAME=command-r-08-2024
```

**Alternative Models:**
- `openai/gpt-4-turbo-preview` (high quality)
- `anthropic/claude-3-haiku` (fast)
- `google/gemini-1.5-flash` (cost-effective)

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Web Application

**Terminal 1 - Backend:**
```bash
cd backend
python main.py
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm install
npm run dev
```

**Access the Application:**
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

### 5. Using the Web Interface

1. **Upload Documents**: Drag and drop PDFs or click to browse
2. **Enter Research Query**: Type your research question
3. **Start Research**: Click button and watch real-time progress
4. **View Results**: See formatted literature review with IEEE-style references

### 6. CLI Mode (Legacy)

```bash
# Basic run (ingest PDFs in data/ then query)
python run_demo.py --query "Machine unlearning in federated learning"

# Skip ingestion if already indexed
python run_demo.py --query "Local Differential Privacy" --skip-ingest

# Export results as Markdown, BibTeX and JSON
python run_demo.py --query "..." --skip-ingest --export

# Add live ArXiv paper search
python run_demo.py --query "..." --skip-ingest --arxiv
```

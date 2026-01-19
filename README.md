# Multi-Agent RAG for Systematic Literature Review

This repository hosts a **Multi-Agent RAG System** designed to perform systematic literature reviews. It goes beyond simple retrieval by orchestration agents to research, extract, screen, and synthesize academic papers.

## 🏗️ Architecture

The system is built on a **Shared Vector Store** model where multiple specialized agents operate:

1.  **Ingestion Agent**: Parses PDFs and indexes them into ChromaDB.
2.  **Retrieval Agents**: Specialized retrievers for different aspects (Methods, Results, etc.).
3.  **Aggregator Agent**: Synthesizes findings from multiple retrievals into a coherent review.

## 🚀 Tech Stack

-   **LLM**: Supports Local LLMs (via Ollama) or API-based.
-   **Vector DB**: ChromaDB (Run locally).
-   **Orchestration**: LangGraph.
-   **Embeddings**: Sentence-Transformers (HuggingFace).

## 📂 Project Structure

```
multi_agent_rag_lit_review/
├── agents/                 # Agent logic (Research, Screening, etc.)
├── rag/                    # RAG pipeline (Ingest, Embed, Retrieve)
├── orchestration/          # LangGraph workflows
├── app/                    # API / Frontend entry points
├── notebooks/              # Experiments & POCs
└── data/                   # Local storage for PDFs & ChromaDB
```

## ⚡ Quick Start

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run POC**:
    Open `notebooks/poc.ipynb` to see a minimal working example.

# Multi-Agent RAG for Systematic Literature Review

This repository hosts a **Multi-Agent RAG System** designed to perform systematic literature reviews. It goes beyond simple retrieval by orchestration agents to research, extract, screen, and synthesize academic papers.

## 🏗️ Architecture

The system is built on a **Shared Vector Store** model where multiple specialized agents operate:

1.  **Ingestion Agent**: Parses PDFs, chunks text, and indexes them into a vector store.
2.  **Query Expansion Agent**: Uses an LLM to generate diverse, targeted search queries from a single research topic.
3.  **Retrieval Agents**: Fetches relevant context using cosine similarity.
4.  **Aggregator Agent**: Synthesizes findings from multiple retrievals into a coherent, cited literature review using an LLM.

## 🚀 Tech Stack

-   **LLM**: Supports Local LLMs (via Ollama, specifically `llama3`) or API-based.
-   **Vector Store**: Lightweight, in-memory vector store with JSON persistence (no heavy C++ dependencies).
-   **Orchestration**: LangGraph.
-   **Embeddings**: Sentence-Transformers (HuggingFace).

## 📂 Project Structure

```
multi_agent_rag_lit_review/
├── agents/                 # Agent logic (Query Expansion, Aggregation)
├── rag/                    # RAG pipeline (Ingest, Embed, Retrieve, Index)
├── orchestration/          # LangGraph workflows
├── app/                    # API / Frontend entry points
├── notebooks/              # Experiments & POCs
└── data/                   # Local storage for PDFs & Index
```

## ⚡ Quick Start

1.  **Prerequisite**: Install [Ollama](https://ollama.com) and pull the model:
    ```bash
    ollama pull llama3
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run POC**:
    Open `notebooks/poc.ipynb` to see the agents in action.

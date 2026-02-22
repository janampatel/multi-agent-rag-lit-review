# 🗺️ Master Implementation Plan: Multi-Agent RAG Evolution

This document outlines the phased roadmap for transforming the current proof-of-concept into a production-grade Multi-Agent RAG system for Systematic Literature Reviews.

---

## 📋 Phase Overview

| Phase | Branch Name | Focus | Items Covered |
|---|---|---|---|
| **1** | `fix/critical-bugs-and-cli` | Bug fixes + usability | #1, #4, #6 |
| **2** | `feat/rag-pipeline-upgrade` | Core RAG quality | #2, #7, #3, #8 |
| **3** | `feat/caching-and-export` | Persistence + outputs | #5, #10 |
| **4** | `feat/multi-agent-orchestration` | True multi-agent graph | Part 2 (Supervisor, Parallel, Critic) |
| **5** | `feat/external-sources-and-eval` | External tools + quality | #13, #14, #15 |

---

## 🔴 Phase 1 — `fix/critical-bugs-and-cli` (DONE ✅)
> **Goal**: Make the existing code actually work correctly and be usable.

- **Commit 1**: Add missing typing imports (`List`, `Dict`, `TypedDict`) to `graph.py`.
- **Commit 2**: Fix ID mismatch bug in `ScreeningAgent` (casting `kid` to `int`).
- **Commit 3**: Replace hardcoded query with `argparse` support in `run_demo.py`.
- **Commit 4**: Cleanup `requirements.txt` and add initial future dependencies.
- **Commit 5**: Update `README.md` with CLI usage.

---

## 🟡 Phase 2 — `feat/rag-pipeline-upgrade`
> **Goal**: Replace toy RAG components with production-grade ones.

- **Commit 1**: Add deduplication check on ingestion using content hashing in `index.py`.
- **Commit 2**: Extract rich PDF metadata (Title, Authors, Year) in `ingest.py` using `PyMuPDF`.
- **Commit 3**: Replace custom JSON vector store with real **ChromaDB** HNSW indexing in `index.py`.
- **Commit 4**: Add a **Cross-Encoder Re-ranker** node (`rag/reranker.py`) to the graph to improve precision after retrieval.

---

## 🟢 Phase 3 — `feat/caching-and-export`
> **Goal**: Add response caching and structured output export.

- **Commit 1**: Add LLM response caching using `diskcache` to save on Ollama compute/latency.
- **Commit 2**: Create `utils/exporter.py` to save results as **Markdown**, **BibTeX**, and **JSON reports**.
- **Commit 3**: Configure `.gitignore` to handle `outputs/` and `.cache/`.

---

## 🔵 Phase 4 — `feat/multi-agent-orchestration`
> **Goal**: Transform the linear chain into a true multi-agent graph with autonomy.

- **Commit 1**: Expand `AgentState` to support planning, voting, and section synthesis.
- **Commit 2**: Add a **Supervisor Agent** node that generates a research plan and acts as a quality gate.
- **Commit 3**: Implement a **Screening Council** (3 parallel LLM screeners) with a voting mechanism (e.g., 2/3 majorities).
- **Commit 4**: Implement **Parallel Section Writers** (Methods, Results, Future Work) to generate the draft simultaneously.
- **Commit 5**: Add a **Critic Agent** node that checks for hallucinations and routes back to synthesis for revisions if needed.
- **Commit 6**: Implement conditional retry loops if the Supervisor finds insufficient relevant papers.

---

## 🟣 Phase 5 — `feat/external-sources-and-eval`
> **Goal**: Add live discovery, human oversight, and quality metrics.

- **Commit 1**: Add an **ArXiv Search Agent** for live paper discovery alongside local PDF retrieval.
- **Commit 2**: Implement a **Human-in-the-Loop** checkpoint using LangGraph's `interrupt_before` synthesis.
- **Commit 3**: Create an **Evaluation Pipeline** (RAGAS-style) to measure Faithfulness, Relevance, and Precision.
- **Commit 4**: Final documentation and performance optimization.

---

## 🏛️ Desired Architecture (Part 2 Brainstorm)

The system will evolve from a sequential chain to a **Supervisor-led Research Council**:

1. **Supervisor**: Decomposes query → Sets Research Plan.
2. **Retrieval Fan-out**: Local RAG + ArXiv + Citation Chasing in parallel.
3. **Synthesis Fan-out**: Separate agents writing distinct sections of the Lit Review.
4. **Validation Loop**: Critic checks Synthesis vs Evidence → Loops back if logic or evidence is flawed.

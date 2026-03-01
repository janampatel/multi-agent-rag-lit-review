
import os
import sys
import argparse
import fitz  # PyMuPDF
from dotenv import load_dotenv

load_dotenv('.env')
sys.path.append(os.getcwd())

from rag.ingest import load_pdf, chunk_text
from rag.embed import embed_texts
from rag.index import VectorStore
from orchestration.graph import app
from utils.exporter import export_all
from utils.evaluator import RAGEvaluator


def ensure_data_directory(data_dir: str):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created {data_dir} directory.")


def create_sample_pdf(path: str):
    if not os.path.exists(path):
        print(f"Generating sample PDF at {path}...")
        doc = fitz.open()
        page = doc.new_page()
        text = """
        Machine unlearning in federated learning (FL) involves removing a client's contribution without retraining the global model from scratch.
        Exact unlearning is computationally expensive because it requires retraining.
        Approximate unlearning uses techniques like gradient ascent to reverse the learning process.

        Key challenges include:
        1. Verification: How to prove data was unlearned?
        2. Efficiency: It must be faster than retraining.
        3. Accuracy: The model performance on remaining data should not degrade significantly.
        """
        page.insert_text((50, 50), text)
        doc.save(path)
        print("Created dummy PDF.")
    else:
        print(f"Sample PDF found at {path}")


def ingest_data(pdf_path: str, persist_directory: str):
    print(f"Starting ingestion process for {pdf_path}...")
    pages = load_pdf(pdf_path)
    print(f"Loaded {len(pages)} pages.")
    chunks = chunk_text(pages)
    print(f"Created {len(chunks)} text chunks.")
    chunk_texts_list = [c['page_content'] for c in chunks]
    print("Generating embeddings (this might take a moment if downloading model)...")
    embeddings = embed_texts(chunk_texts_list)
    print("Indexing into storage...")
    vs = VectorStore(persist_directory=persist_directory)
    vs.add_documents(chunks, embeddings)
    print("Ingestion complete!")


def fetch_arxiv_docs(query: str, max_results: int = 5):
    """Optionally fetch ArXiv papers for the query."""
    try:
        from agents.web_search import ArxivSearchAgent
        agent = ArxivSearchAgent()
        return agent.search(query, max_results=max_results)
    except Exception as e:
        print(f"[ArXiv] Failed: {e}")
        return []


def human_review_checkpoint(screened_docs):
    """
    Pause after screening and let the user review/edit the paper list.
    Returns the (possibly modified) list.
    """
    print("\n--- HUMAN REVIEW CHECKPOINT ---")
    print(f"The council screened {len(screened_docs)} papers:\n")
    for i, doc in enumerate(screened_docs, 1):
        meta  = doc.get("metadata", {})
        title = meta.get("title", "Unknown")[:70]
        year  = meta.get("year", "?")
        print(f"  [{i}] {title} ({year})")

    print("\nOptions: 'yes' to proceed | 'abort' to stop | comma-separated IDs to remove (e.g. '2,4')")
    try:
        choice = input("Your choice: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        choice = "yes"

    if choice == "abort":
        print("Aborted by user.")
        sys.exit(0)
    elif choice != "yes":
        try:
            remove_ids = {int(x.strip()) - 1 for x in choice.split(",") if x.strip().isdigit()}
            screened_docs = [d for i, d in enumerate(screened_docs) if i not in remove_ids]
            print(f"Removed {len(remove_ids)} paper(s). {len(screened_docs)} remaining.")
        except Exception:
            print("Could not parse IDs — proceeding with all papers.")

    print("--- END HUMAN REVIEW ---\n")
    return screened_docs


def run_agent_workflow(
    query: str,
    export: bool = False,
    use_arxiv: bool = False,
    human_review: bool = False,
    quality_threshold: int = 3,
):
    print("\n--- Running Multi-Agent RAG System ---\n")
    print(f"Query: {query}")

    # Optionally fetch ArXiv docs before running the graph
    web_docs = []
    if use_arxiv:
        print("[ArXiv] Searching live papers...")
        web_docs = fetch_arxiv_docs(query)

    initial_state = {
        "query":              query,
        "expanded_queries":   [],
        "web_search_docs":    web_docs,
        "local_docs":         [],
        "merged_docs":        [],
        "retrieved_docs":     [],
        "retrieval_attempts": 0,
        "quality_threshold":  quality_threshold,
        "screened_docs":      [],
        "section_methods":    "",
        "section_results":    "",
        "section_challenges": "",
        "draft_review":       "",
        "critic_issues":      [],
        "revision_count":     0,
        "final_response":     "",
        "exported_files":     [],
    }

    try:
        # Run the full graph
        result = app.invoke(initial_state)

        # Human-in-the-loop checkpoint (post-screening, pre-synthesis)
        # Note: Because LangGraph runs the full graph in one invoke(), we do this
        # as a post-processing check on the screened_docs in the result, then
        # re-run synthesis if the user edits the list.
        if human_review and result.get("screened_docs"):
            edited_docs = human_review_checkpoint(result["screened_docs"])
            if edited_docs != result["screened_docs"]:
                # Re-synthesize with the edited set
                print("[Human Review] Re-running synthesis with edited paper set...")
                from agents.section_writers import (
                    MethodsSectionWriter, ResultsSectionWriter,
                    ChallengesSectionWriter, merge_sections,
                )
                from agents.critic import CriticAgent
                methods    = MethodsSectionWriter().write(edited_docs, query)
                results_s  = ResultsSectionWriter().write(edited_docs, query)
                challenges = ChallengesSectionWriter().write(edited_docs, query)
                draft      = merge_sections(methods, results_s, challenges, query)
                issues     = CriticAgent().critique(draft, edited_docs)
                result["screened_docs"]  = edited_docs
                result["retrieved_docs"] = edited_docs
                result["draft_review"]   = draft
                result["final_response"] = draft
                result["critic_issues"]  = issues
                result["revision_count"] = result.get("revision_count", 0)

        print("\n--- FINAL SYNTHESIZED RESPONSE ---\n")
        print(result.get("final_response", "No response generated."))
        print("\n----------------------------------")

        # Evaluation report
        evaluator = RAGEvaluator()
        report = evaluator.generate_report(result)
        evaluator.print_report(report)

        # Export
        if export:
            paths = export_all(
                query=query,
                review_text=result.get("final_response", ""),
                docs=result.get("retrieved_docs", []),
            )
            # Attach eval report into the JSON export
            import json
            for p in paths.values():
                if p.endswith(".json"):
                    try:
                        with open(p, "r", encoding="utf-8") as f:
                            payload = json.load(f)
                        payload["evaluation"] = report
                        with open(p, "w", encoding="utf-8") as f:
                            json.dump(payload, f, indent=2, ensure_ascii=False)
                    except Exception:
                        pass

    except Exception as e:
        print(f"\nError running the agent workflow: {e}")
        import traceback; traceback.print_exc()
        print("\nTroubleshooting:")
        print("1. Ensure BACKBOARD_API_KEY is set in .env")
        print("2. Run 'python test_backboard.py' to verify connection")
        print("3. Check that your Backboard account has access to the configured model")


def main():
    parser = argparse.ArgumentParser(description="Multi-Agent RAG for Literature Review")
    parser.add_argument("--query",      type=str, default="Methods for unlearning in FL",
                        help="Research question")
    parser.add_argument("--data-dir",   type=str, default="data",
                        help="Directory containing PDFs to ingest")
    parser.add_argument("--persist-dir", type=str, default="data/faiss_index",
                        help="Directory for vector store persistence")
    parser.add_argument("--skip-ingest", action="store_true",
                        help="Skip ingestion phase")
    parser.add_argument("--export",     action="store_true", default=True,
                        help="Export results to outputs/ as Markdown, BibTeX and JSON")
    parser.add_argument("--arxiv",      action="store_true", default=True,
                        help="Fetch live ArXiv papers for the query")
    parser.add_argument("--human-review", action="store_true",
                        help="Pause after screening for human review of paper list")
    parser.add_argument("--threshold",  type=int, default=3,
                        help="Minimum screened docs needed before synthesis (default 3)")

    args = parser.parse_args()

    ensure_data_directory(args.data_dir)

    if not args.skip_ingest:
        pdf_files = [f for f in os.listdir(args.data_dir) if f.lower().endswith(".pdf")]
        if not pdf_files:
            print(f"No PDFs found in {args.data_dir}/. Creating sample...")
            sample_path = os.path.join(args.data_dir, "sample_paper.pdf")
            create_sample_pdf(sample_path)
            pdf_files = ["sample_paper.pdf"]

        print(f"Found {len(pdf_files)} PDF(s) to ingest: {pdf_files}")
        for pdf_file in pdf_files:
            ingest_data(os.path.join(args.data_dir, pdf_file), args.persist_dir)
    else:
        print("Skipping ingestion phase as requested.")

    run_agent_workflow(
        query=args.query,
        export=args.export,
        use_arxiv=args.arxiv,
        human_review=args.human_review,
        quality_threshold=args.threshold,
    )


if __name__ == "__main__":
    main()

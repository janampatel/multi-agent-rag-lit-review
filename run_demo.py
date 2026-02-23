
import os
import sys
import argparse
import fitz  # PyMuPDF
from dotenv import load_dotenv

# Load environment variables
load_dotenv('.env')

# Ensure we can import from the project root
sys.path.append(os.getcwd())

from rag.ingest import load_pdf, chunk_text
from rag.embed import embed_texts
from rag.index import VectorStore
from orchestration.graph import app
from utils.exporter import export_all

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
    # Load
    pages = load_pdf(pdf_path)
    print(f"Loaded {len(pages)} pages.")
    
    # Chunk
    chunks = chunk_text(pages)
    print(f"Created {len(chunks)} text chunks.")
    
    # Embed
    chunk_texts = [c['page_content'] for c in chunks]
    print("Generating embeddings (this might take a moment if downloading model)...")
    embeddings = embed_texts(chunk_texts)
    
    # Index
    print("Indexing into storage...")
    vs = VectorStore(persist_directory=persist_directory)
    vs.add_documents(chunks, embeddings)
    print("Ingestion complete!")

def run_agent_workflow(query: str, export: bool = False):
    initial_state = {
        "query": query,
        "expanded_queries": [],
        "retrieved_docs": [],
        "final_response": ""
    }
    
    print("\n--- Running Multi-Agent RAG System ---\n")
    print(f"Initial Query: {initial_state['query']}")
    
    try:
        result = app.invoke(initial_state)
        
        print("\n--- FINAL SYNTHESIZED RESPONSE ---\n")
        print(result['final_response'])
        print("\n----------------------------------")

        if export:
            export_all(
                query=query,
                review_text=result['final_response'],
                docs=result['retrieved_docs'],
            )
        
    except Exception as e:
        print(f"\nError running the agent workflow: {e}")
        print("Ensure that Ollama is running and the model set in .env is pulled.")
        print("Currently using model:", os.getenv('OLLAMA_MODEL', 'llama2'))
        print("To pull a different model: 'ollama pull <model_name>'")

def main():
    parser = argparse.ArgumentParser(description="Multi-Agent RAG for Literature Review")
    parser.add_argument("--query", type=str, default="Methods for unlearning in FL",
                        help="Research question to investigate")
    parser.add_argument("--data-dir", type=str, default="data",
                        help="Directory containing PDFs to ingest")
    parser.add_argument("--skip-ingest", action="store_true",
                        help="Skip ingestion phase")
    parser.add_argument("--persist-dir", type=str, default="data/chroma_db",
                        help="Directory for vector store persistence")
    parser.add_argument("--export", action="store_true",
                        help="Export results to outputs/ as Markdown, BibTeX and JSON")
    
    args = parser.parse_args()
    
    ensure_data_directory(args.data_dir)
    
    if not args.skip_ingest:
        # Find all PDFs in data_dir
        pdf_files = [f for f in os.listdir(args.data_dir) if f.lower().endswith(".pdf")]
        
        if not pdf_files:
            print(f"No PDFs found in {args.data_dir}/. Creating sample...")
            sample_path = os.path.join(args.data_dir, "sample_paper.pdf")
            create_sample_pdf(sample_path)
            pdf_files = ["sample_paper.pdf"]
        
        print(f"Found {len(pdf_files)} PDF(s) to ingest: {pdf_files}")
        
        # Ingest each
        for pdf_file in pdf_files:
            full_path = os.path.join(args.data_dir, pdf_file)
            ingest_data(full_path, args.persist_dir)
    else:
        print("Skipping ingestion phase as requested.")
    
    run_agent_workflow(args.query, export=args.export)

if __name__ == "__main__":
    main()


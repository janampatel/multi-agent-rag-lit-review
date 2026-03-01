import os
import sys
import uuid
from typing import Dict, List
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestration.graph import app as research_app
from rag.ingest import load_pdf, chunk_text
from rag.embed import embed_texts
from rag.index import VectorStore

app = FastAPI(title="Multi-Agent RAG API")

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for MVP
jobs: Dict[str, dict] = {}
documents: List[dict] = []

class ResearchQuery(BaseModel):
    query: str
    use_arxiv: bool = False
    threshold: int = 3

@app.get("/")
async def root():
    return {"message": "Multi-Agent RAG API is running"}

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "documents": len(documents)}

@app.post("/api/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files allowed")
    
    # Save uploaded file
    file_path = f"../data/{file.filename}"
    os.makedirs("../data", exist_ok=True)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Process PDF
    try:
        pages = load_pdf(file_path)
        chunks = chunk_text(pages)
        chunk_texts = [c['page_content'] for c in chunks]
        embeddings = embed_texts(chunk_texts)
        
        vs = VectorStore(persist_directory="../data/faiss_index")
        vs.add_documents(chunks, embeddings)
        
        doc_info = {
            "id": str(uuid.uuid4()),
            "filename": file.filename,
            "pages": len(pages),
            "chunks": len(chunks),
            "status": "processed"
        }
        documents.append(doc_info)
        
        return {"message": "Document uploaded and processed", "document": doc_info}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.get("/api/documents")
async def list_documents():
    return {"documents": documents}

@app.post("/api/research/start")
async def start_research(query: ResearchQuery, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    
    jobs[job_id] = {
        "job_id": job_id,
        "status": "running",
        "progress": 0.0,
        "current_step": "Starting research...",
        "query": query.query,
        "result": "",
        "error": ""
    }
    
    background_tasks.add_task(run_research_workflow, job_id, query)
    return {"job_id": job_id}

@app.get("/api/research/status/{job_id}")
async def get_job_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]

async def run_research_workflow(job_id: str, query: ResearchQuery):
    try:
        jobs[job_id].update({
            "progress": 0.1,
            "current_step": "Expanding query into variations..."
        })
        
        initial_state = {
            "query": query.query,
            "expanded_queries": [],
            "web_search_docs": [],
            "local_docs": [],
            "merged_docs": [],
            "retrieved_docs": [],
            "retrieval_attempts": 0,
            "quality_threshold": query.threshold,
            "screened_docs": [],
            "section_methods": "",
            "section_results": "",
            "section_challenges": "",
            "draft_review": "",
            "critic_issues": [],
            "revision_count": 0,
            "final_response": "",
            "exported_files": [],
        }
        
        jobs[job_id].update({
            "progress": 0.2,
            "current_step": "Retrieving relevant documents..."
        })
        
        import time
        time.sleep(1)  # Brief pause for UI feedback
        
        jobs[job_id].update({
            "progress": 0.4,
            "current_step": "Screening papers with AI council..."
        })
        
        result = research_app.invoke(initial_state)
        
        jobs[job_id].update({
            "progress": 0.9,
            "current_step": "Adding references..."
        })
        
        # Add proper references
        from agents.section_writers import add_references
        final_text = result.get("final_response", "No response generated")
        docs = result.get("retrieved_docs", [])  # Use retrieved_docs instead of screened_docs
        
        print(f"[Backend] Found {len(docs)} documents for references")
        
        if docs and len(docs) > 0:
            final_text = add_references(final_text, docs)
            print(f"[Backend] References added successfully")
        else:
            print(f"[Backend] Warning: No documents found for references")
        
        jobs[job_id].update({
            "status": "completed",
            "progress": 1.0,
            "current_step": "Research completed!",
            "result": final_text
        })
        
    except Exception as e:
        jobs[job_id].update({
            "status": "failed",
            "progress": 0.0,
            "current_step": "Failed",
            "error": str(e)
        })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
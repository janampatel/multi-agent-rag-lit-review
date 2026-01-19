
import os
import fitz  # PyMuPDF
from typing import List, Dict

def load_pdf(file_path: str) -> List[Dict]:
    """
    Reads a PDF file and returns a list of page objects.
    Each object contains 'page_content' and 'metadata'.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    doc = fitz.open(file_path)
    pages = []
    
    for page_num, page in enumerate(doc):
        text = page.get_text()
        pages.append({
            "page_content": text,
            "metadata": {
                "source": file_path,
                "page": page_num + 1
            }
        })
    
    return pages

def chunk_text(pages: List[Dict], chunk_size: int = 1000, overlap: int = 200) -> List[Dict]:
    """
    Simple chunking strategy. 
    In a real system, we'd use RecursiveCharacterTextSplitter or Semantic Chunking.
    """
    chunks = []
    for page in pages:
        text = page["page_content"]
        metadata = page["metadata"]
        
        # Very naive splitting for MVP
        for i in range(0, len(text), chunk_size - overlap):
            chunk_text = text[i:i + chunk_size]
            chunks.append({
                "page_content": chunk_text,
                "metadata": metadata
            })
            
    return chunks

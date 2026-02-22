
import os
import re
import fitz  # PyMuPDF
from typing import List, Dict


def _parse_year(raw_date: str) -> str:
    """
    Extracts a 4-digit year from a PDF creation date string.
    PDF dates are typically in the format 'D:YYYYMMDDHHmmSS...'
    Falls back to regex search for any 4-digit year-like sequence.
    """
    # Standard PDF date format: D:20230115...
    if raw_date.startswith("D:") and len(raw_date) >= 6:
        candidate = raw_date[2:6]
        if candidate.isdigit():
            return candidate

    # Fallback: find any 4-digit number between 1900 and 2099
    match = re.search(r"(19|20)\d{2}", raw_date)
    if match:
        return match.group(0)

    return "Unknown"


def load_pdf(file_path: str) -> List[Dict]:
    """
    Reads a PDF and returns a list of page dicts, each with
    'page_content' and enriched 'metadata' (title, authors, year, etc.).
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    doc = fitz.open(file_path)
    pdf_meta = doc.metadata  # {'title': ..., 'author': ..., 'creationDate': ...}

    title = pdf_meta.get("title", "").strip() or os.path.splitext(os.path.basename(file_path))[0]
    authors = pdf_meta.get("author", "").strip() or "Unknown Authors"
    year = _parse_year(pdf_meta.get("creationDate", ""))

    pages = []
    for page_num, page in enumerate(doc):
        text = page.get_text()
        if not text.strip():  # Skip blank pages
            continue
        pages.append({
            "page_content": text,
            "metadata": {
                "source": file_path,
                "filename": os.path.basename(file_path),
                "page": page_num + 1,
                "total_pages": len(doc),
                "title": title,
                "authors": authors,
                "year": year,
            }
        })

    doc.close()
    return pages


def chunk_text(pages: List[Dict], chunk_size: int = 1000, overlap: int = 200) -> List[Dict]:
    """
    Splits page text into overlapping chunks.
    Attempts to break on sentence boundaries ('. ', '? ', '! ') to avoid
    cutting mid-sentence, falling back to hard character splits.
    """
    sentence_endings = re.compile(r'(?<=[.?!])\s+')
    chunks = []

    for page in pages:
        text = page["page_content"]
        metadata = page["metadata"]

        # Split text into sentences first, then recombine into chunks
        sentences = sentence_endings.split(text)
        current_chunk = ""

        for sentence in sentences:
            # If adding this sentence stays within chunk_size, append it
            if len(current_chunk) + len(sentence) + 1 <= chunk_size:
                current_chunk = (current_chunk + " " + sentence).strip()
            else:
                # Save the current chunk if it has content
                if current_chunk:
                    chunks.append({
                        "page_content": current_chunk,
                        "metadata": metadata
                    })
                # Start a new chunk with overlap: carry the tail of the last chunk
                overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                current_chunk = (overlap_text + " " + sentence).strip()

        # Don't forget the last chunk
        if current_chunk:
            chunks.append({
                "page_content": current_chunk,
                "metadata": metadata
            })

    return chunks

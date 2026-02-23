
import os
import re
import json
import urllib.request
import urllib.error
import fitz  # PyMuPDF
from typing import List, Dict, Tuple, Optional


# ---------------------------------------------------------------------------
# DOI-based metadata lookup (Crossref API — free, no key required)
# ---------------------------------------------------------------------------

_DOI_REGEX   = re.compile(r"\b10\.\d{4,}/[^\s\"'<>]+", re.IGNORECASE)
_ARXIV_REGEX = re.compile(r"arXiv[:\s]+(\d{4}\.\d{4,5})", re.IGNORECASE)


def _find_doi(text: str) -> Optional[str]:
    """Extracts the first DOI-like pattern found in any text string."""
    match = _DOI_REGEX.search(text)
    if match:
        return match.group(0).rstrip(".,;)")
    return None


def _find_arxiv_id(text: str) -> Optional[str]:
    """Extracts an arXiv ID like '2402.08787' from any text string."""
    match = _ARXIV_REGEX.search(text)
    return match.group(1) if match else None


_METADATA_CACHE_PATH = ".cache/paper_metadata.json"


def _load_metadata_cache() -> Dict:
    """Loads the local paper metadata cache from disk."""
    if os.path.exists(_METADATA_CACHE_PATH):
        try:
            with open(_METADATA_CACHE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_metadata_cache(cache: Dict):
    """Persists the local paper metadata cache to disk."""
    os.makedirs(os.path.dirname(_METADATA_CACHE_PATH), exist_ok=True)
    with open(_METADATA_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def _query_semantic_scholar(paper_id: str) -> Optional[Dict]:
    """
    Queries the Semantic Scholar API for paper metadata.
    Results are cached locally in .cache/paper_metadata.json to avoid
    repeated API calls and rate-limit issues (429 Too Many Requests).

    Works for both DOI (prefix 'DOI:') and arXiv ID (prefix 'arXiv:').
    Free API, no key required — 100 req/5min unauthenticated.
    """
    cache = _load_metadata_cache()
    if paper_id in cache:
        print(f"  [S2 cache] {paper_id} → '{cache[paper_id].get('title', '')[:60]}'")
        return cache[paper_id]

    url = (
        f"https://api.semanticscholar.org/graph/v1/paper/{urllib.request.quote(paper_id)}"
        "?fields=title,authors,year"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "MultiAgentRAG/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=6) as resp:
            data = json.loads(resp.read().decode())

        title   = data.get("title", "")
        authors = ", ".join(
            a.get("name", "") for a in data.get("authors", []) if a.get("name")
        )
        year = str(data.get("year", "")) or "Unknown"

        if title:
            result = {"title": title, "authors": authors, "year": year}
            cache[paper_id] = result
            _save_metadata_cache(cache)
            print(f"  [S2] {paper_id} → '{title[:60]}' ({year})")
            return result
    except Exception as e:
        print(f"  [S2] Lookup failed for {paper_id}: {e}")
    return None


def _fetch_metadata_from_doi(doi: str) -> Optional[Dict]:
    """
    Queries the Crossref REST API for structured metadata for a given DOI.
    Returns a dict with 'title', 'authors', 'year' or None on failure.
    """
    # Try Semantic Scholar first (faster + works for both DOI and arXiv)
    result = _query_semantic_scholar(f"DOI:{doi}")
    if result:
        return result

    # Fallback: native Crossref API
    url = f"https://api.crossref.org/works/{urllib.request.quote(doi, safe='/')}"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "MultiAgentRAG/1.0 (mailto:research@example.com)"}
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
        msg = data.get("message", {})

        titles = msg.get("title", [])
        title  = titles[0] if titles else ""

        raw_authors  = msg.get("author", [])
        author_parts = [
            f"{a.get('given', '')} {a.get('family', '')}".strip()
            for a in raw_authors
        ]
        authors = ", ".join(p for p in author_parts if p)

        year = ""
        for date_field in ("published-print", "published-online", "created"):
            parts = msg.get(date_field, {}).get("date-parts", [[]])[0]
            if parts and parts[0]:
                year = str(parts[0])
                break

        if title:
            print(f"  [Crossref] {doi} → '{title[:60]}' ({year})")
            return {"title": title, "authors": authors, "year": year}
    except Exception as e:
        print(f"  [Crossref] Lookup failed for {doi}: {e}")
    return None

def _parse_year_from_string(text: str) -> str:
    """Scan any string for a plausible publication year (1990-2099)."""
    match = re.search(r"\b(19[9]\d|20[0-2]\d)\b", text)
    return match.group(0) if match else "Unknown"


def _parse_year_from_pdf_date(raw_date: str) -> str:
    """
    Extracts a 4-digit year from a PDF creation date string.
    PDF dates are typically in the format 'D:YYYYMMDDHHmmSS...'
    """
    if raw_date.startswith("D:") and len(raw_date) >= 6:
        candidate = raw_date[2:6]
        if candidate.isdigit() and 1900 <= int(candidate) <= 2099:
            return candidate
    return _parse_year_from_string(raw_date)


def _looks_like_authors(line: str) -> bool:
    """
    Heuristic: author lines typically contain comma-separated capitalised words
    or 'and', are not overly long, and don't look like section headings.
    Also handles the common academic format: "John Smith1, Jane Doe2†"
    where superscript digits or symbols (†, *, ‡, §) follow names.
    """
    line = line.strip()
    if not line or len(line) > 400:
        return False
    # Must have at least one comma or ' and '
    if "," not in line and " and " not in line.lower():
        return False
    # Strip superscript indicators before checking for sentence punctuation
    cleaned = re.sub(r"[\d†‡§∗\*]", "", line)
    if re.search(r"[.!?;:]", cleaned):
        return False
    # Most tokens should start with a capital letter
    tokens = [t for t in re.split(r"[\s,]+", cleaned) if t]
    if not tokens:
        return False
    capitalised = sum(1 for t in tokens if t and t[0].isupper())
    return capitalised / len(tokens) >= 0.5


def _extract_metadata_from_plain_text(text: str) -> Tuple[str, str, str]:
    """
    Parses raw first-page text to extract title, authors, and year.

    Strategy (works for most academic papers):
      Title  — first block of non-empty lines that are ALL CAPS or Title Case
               and short enough to be headings (< 120 chars each).
               Stops at the first line that breaks the pattern.
      Authors — the first subsequent line that passes _looks_like_authors().
      Year   — regex scan of the full text.
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    def _is_title_line(ln: str) -> bool:
        """A title line is short and either ALLCAPS or mostly Title-Cased words."""
        if len(ln) > 120:
            return False
        words = ln.split()
        if not words:
            return False
        # Ignore lines that are just numbers or very short codes
        if all(not c.isalpha() for c in ln):
            return False
        upper_ratio = sum(1 for w in words if w.isupper()) / len(words)
        title_ratio = sum(1 for w in words if w and w[0].isupper()) / len(words)
        return upper_ratio >= 0.5 or title_ratio >= 0.7

    # Collect consecutive title lines from the top of the page
    title_lines = []
    for line in lines[:15]:  # Only look in first 15 lines
        if _is_title_line(line):
            title_lines.append(line)
        elif title_lines:
            # Stop at the first non-title line after we have something
            break

    title = " ".join(title_lines).strip() if title_lines else ""

    # Authors: scan lines after the title for the first author-like line
    start_idx = len(title_lines)
    authors = ""
    for line in lines[start_idx: start_idx + 10]:
        if _looks_like_authors(line):
            authors = line
            break

    # Year: regex scan
    year = _parse_year_from_string(text)

    return title, authors, year



def _extract_metadata_from_first_page(page: fitz.Page) -> Tuple[str, str, str]:
    """
    Uses PyMuPDF's structured block/span data to extract title, authors, and year
    from the first page of an academic PDF.

    Strategy:
      Title   — text block(s) rendered in the largest font size on the page,
                with special handling for arXiv/DOI header lines (uses
                second-largest font group when the largest looks like a header).
      Authors — text lines immediately following the title block that look like names.
      Year    — regex search of the full first-page text for a 4-digit year.

    Returns:
        (title, authors, year) — all strings, falling back to "Unknown" each.
    """
    _HEADER_PREFIXES = ("arxiv:", "doi:", "preprint", "submitted to", "accepted to")

    # ---- collect all spans with their font sizes -------------------------
    blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]

    span_data: List[Tuple[float, str, int]] = []  # (font_size, text, block_idx)
    for b_idx, block in enumerate(blocks):
        if block.get("type") != 0:   # 0 = text block
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                txt = span.get("text", "").strip()
                size = span.get("size", 0.0)
                if txt:
                    span_data.append((size, txt, b_idx))

    if not span_data:
        return "Unknown Title", "Unknown Authors", "Unknown"

    # ---- get sorted unique font sizes (descending) ----------------------
    unique_sizes = sorted({s[0] for s in span_data}, reverse=True)

    # ---- try each font-size group until we find a non-header title ------
    title = ""
    title_block_idx = 0
    for font_size in unique_sizes:
        candidate_spans = [s[1] for s in span_data if abs(s[0] - font_size) <= 1.0]
        candidate_text = " ".join(candidate_spans).strip()

        # Skip obvious submission headers (arXiv banner, DOI line, etc.)
        if any(candidate_text.lower().startswith(p) for p in _HEADER_PREFIXES):
            continue
        # Skip very short candidates (unlikely to be a full title)
        if len(candidate_text) < 8:
            continue

        title = candidate_text
        title_block_idx = next(
            (s[2] for s in span_data if abs(s[0] - font_size) <= 1.0), 0
        )
        break

    if not title:
        title = "Unknown Title"

    # Trim to a reasonable length
    if len(title) > 250:
        title = title[:250].rsplit(" ", 1)[0] + "…"

    # ---- authors: scan lines after the title block, looking for name patterns
    authors = "Unknown Authors"
    for block in blocks[title_block_idx + 1: title_block_idx + 8]:
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            line_text = " ".join(
                span.get("text", "") for span in line.get("spans", [])
            ).strip()
            if _looks_like_authors(line_text):
                authors = line_text
                break
        if authors != "Unknown Authors":
            break

    # ---- year: regex scan of the full first page text -------------------
    full_text = page.get_text()
    year = _parse_year_from_string(full_text)

    return title, authors, year


def load_pdf(file_path: str) -> List[Dict]:
    """
    Reads a PDF and returns a list of page dicts, each with
    'page_content' and enriched 'metadata' (title, authors, year, etc.).

    Metadata priority:
      1. PDF embedded properties (doc.metadata) — most reliable when present.
      2. Content-based extraction from page 1 (font size → title, heuristics → authors).
      3. Filename stem as last-resort title fallback.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    doc = fitz.open(file_path)
    pdf_meta = doc.metadata  # may be empty for many academic PDFs
    first_page = doc[0] if len(doc) > 0 else None
    first_page_text = first_page.get_text() if first_page else ""

    title, authors, year = "", "", "Unknown"

    # ---------------------------------------------------------------
    # Priority 1: DOI / arXiv ID → Semantic Scholar API
    # ---------------------------------------------------------------
    doi_candidate   = (
        _find_doi(pdf_meta.get("doi", ""))
        or _find_doi(pdf_meta.get("subject", ""))
        or _find_doi(first_page_text[:3000])
    )
    arxiv_candidate = _find_arxiv_id(first_page_text[:1000])

    if doi_candidate:
        crossref = _fetch_metadata_from_doi(doi_candidate)
        if crossref:
            title   = crossref.get("title", "")
            authors = crossref.get("authors", "")
            year    = crossref.get("year", "Unknown")

    if not title and arxiv_candidate:
        s2 = _query_semantic_scholar(f"arXiv:{arxiv_candidate}")
        if s2:
            title   = s2.get("title", "")
            authors = s2.get("authors", "")
            year    = s2.get("year", "Unknown")

    # ---------------------------------------------------------------
    # Priority 2: PDF embedded properties
    # ---------------------------------------------------------------
    if not title:
        title = pdf_meta.get("title", "").strip()
    if not authors:
        authors = pdf_meta.get("author", "").strip()
    if year == "Unknown":
        year = _parse_year_from_pdf_date(pdf_meta.get("creationDate", ""))

    # ---------------------------------------------------------------
    # Priority 3: Plain-text line parsing from first page
    #   (More reliable than font-size for most academic PDFs)
    # ---------------------------------------------------------------
    if first_page and (not title or not authors or year == "Unknown"):
        t, a, y = _extract_metadata_from_plain_text(first_page_text)
        if not title and t:
            title = t
        if not authors and a:
            authors = a
        if year == "Unknown" and y != "Unknown":
            year = y

    # ---------------------------------------------------------------
    # Priority 4: Font-size heuristics (for uncommon layouts)
    # ---------------------------------------------------------------
    if first_page and (not title or not authors):
        t, a, y = _extract_metadata_from_first_page(first_page)
        if not title and t not in ("Unknown Title", ""):
            title = t
        if not authors and a not in ("Unknown Authors", ""):
            authors = a
        if year == "Unknown" and y != "Unknown":
            year = y

    # ---------------------------------------------------------------
    # Priority 4: Filename stem (last resort)
    # ---------------------------------------------------------------
    if not title:
        title = os.path.splitext(os.path.basename(file_path))[0]
    if not authors:
        authors = "Unknown Authors"

    print(f"  Metadata → Title: '{title[:70]}' | Authors: '{authors[:50]}' | Year: {year}")

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

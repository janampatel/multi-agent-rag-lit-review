"""
agents/web_search.py — ArXiv live search agent.

Returns results in the same {content, metadata} format as the local retriever
so they can be merged seamlessly into the retrieval pool.
"""

from typing import List, Dict


class ArxivSearchAgent:
    """
    Searches ArXiv live for papers matching the research topic.
    Uses the official `arxiv` Python client.
    Activated only when --arxiv flag is passed (opt-in).
    """

    def search(self, query: str, max_results: int = 5) -> List[Dict]:
        """
        Args:
            query:       Research topic string.
            max_results: Maximum number of papers to fetch (default 5 for speed).

        Returns:
            List of {content, metadata} dicts identical to local retriever format.
        """
        try:
            import arxiv  # Lazy import so missing package gives a clear error

            client = arxiv.Client()
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance,
            )

            results = []
            for paper in client.results(search):
                results.append({
                    "content": paper.summary,
                    "metadata": {
                        "title":   paper.title,
                        "authors": ", ".join(a.name for a in paper.authors),
                        "year":    str(paper.published.year),
                        "source":  paper.entry_id,
                        "url":     paper.pdf_url,
                        "doi":     paper.doi or "N/A",
                        "filename": f"arxiv_{paper.entry_id.split('/')[-1]}.pdf",
                    }
                })

            print(f"[ArXiv] Found {len(results)} papers for: '{query}'")
            return results

        except ImportError:
            print("[ArXiv] 'arxiv' package not installed. Run: pip install arxiv")
            return []
        except Exception as e:
            print(f"[ArXiv] Search failed: {e}")
            return []

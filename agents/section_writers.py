"""
agents/section_writers.py — Focused section writers for the literature review.

Each writer produces one section of the review using a short, focused prompt.
Backboard handles caching and provides faster inference than local Ollama.
"""

import os
from typing import List, Dict
from dotenv import load_dotenv
from utils.backboard_langchain import BackboardLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


def _build_evidence(docs: List[Dict], max_chars: int = 2000) -> str:
    """Formats the source documents into a compact evidence string with metadata."""
    evidence = ""
    for i, doc in enumerate(docs, 1):
        meta = doc.get("metadata", {})
        title = meta.get("title", meta.get("filename", "Unknown"))
        authors = meta.get("authors", "Unknown Authors")
        year = meta.get("year", "Unknown Year")
        snippet = doc.get("content", "")[:200].replace("\n", " ")
        evidence += f"[{i}] {title} ({authors}, {year}): {snippet}...\n"
        if len(evidence) >= max_chars:
            break
    return evidence


def _make_chain():
    llm = BackboardLLM(temperature=0.3, use_memory=False)
    prompt = PromptTemplate(
        template="{instruction}\n\nResearch Topic: {query}\n\nEvidence:\n{evidence}\n\nSection:",
        input_variables=["instruction", "query", "evidence"],
    )
    return prompt | llm | StrOutputParser()


class MethodsSectionWriter:
    """Writes the 'Methods & Approaches' section of the literature review."""

    INSTRUCTION = (
        "Write a concise 'Methods & Approaches' section for a literature review. "
        "Summarize the key methodologies described in the evidence. "
        "Use ONLY the citations [1], [2], etc. that match the evidence provided. "
        "Do NOT make up references. Be factual. Max 250 words."
    )

    def write(self, docs: List[Dict], query: str) -> str:
        chain = _make_chain()
        evidence = _build_evidence(docs)
        return chain.invoke({
            "instruction": self.INSTRUCTION,
            "query": query,
            "evidence": evidence,
        })


class ResultsSectionWriter:
    """Writes the 'Results & Benchmarks' section of the literature review."""

    INSTRUCTION = (
        "Write a concise 'Results & Benchmarks' section for a literature review. "
        "Summarize empirical results, metrics, and comparisons from the evidence. "
        "Use ONLY the citations [1], [2], etc. that match the evidence provided. "
        "Do NOT make up references. Be factual. Max 250 words."
    )

    def write(self, docs: List[Dict], query: str) -> str:
        chain = _make_chain()
        evidence = _build_evidence(docs)
        return chain.invoke({
            "instruction": self.INSTRUCTION,
            "query": query,
            "evidence": evidence,
        })


class ChallengesSectionWriter:
    """Writes the 'Open Challenges & Future Work' section of the literature review."""

    INSTRUCTION = (
        "Write a concise 'Open Challenges & Future Work' section for a literature review. "
        "Identify limitations and open problems from the evidence. "
        "Use ONLY the citations [1], [2], etc. that match the evidence provided. "
        "Do NOT make up references. Be factual. Max 250 words."
    )

    def write(self, docs: List[Dict], query: str) -> str:
        chain = _make_chain()
        evidence = _build_evidence(docs)
        return chain.invoke({
            "instruction": self.INSTRUCTION,
            "query": query,
            "evidence": evidence,
        })


def merge_sections(methods: str, results: str, challenges: str, query: str) -> str:
    """Combines the three section strings into a single draft review."""
    return (
        f"# Literature Review: {query}\n\n"
        f"## Methods & Approaches\n\n{methods.strip()}\n\n"
        f"## Results & Benchmarks\n\n{results.strip()}\n\n"
        f"## Open Challenges & Future Work\n\n{challenges.strip()}\n"
    )


def add_references(review: str, docs: List[Dict]) -> str:
    """Appends a References section in IEEE format with actual document metadata."""
    refs = "\n\n## References\n\n"
    
    # Deduplicate by filename to avoid duplicate references
    seen_files = set()
    unique_docs = []
    for doc in docs:
        filename = doc.get("metadata", {}).get("filename", "")
        if filename and filename not in seen_files:
            seen_files.add(filename)
            unique_docs.append(doc)
    
    for i, doc in enumerate(unique_docs, 1):
        meta = doc.get("metadata", {})
        title = meta.get("title", meta.get("filename", "Unknown"))
        authors = meta.get("authors", "Unknown Authors")
        year = meta.get("year", "Unknown")
        filename = meta.get("filename", "")
        
        # Clean up title - remove ellipsis and truncate if too long
        title = title.replace("…", "").strip()
        if len(title) > 150:
            title = title[:150].rsplit(" ", 1)[0] + "..."
        
        # Format in IEEE style: [N] Authors, "Title," Filename, Year.
        refs += f'[{i}] {authors}, "{title}," {filename}, {year}.\n\n'
    
    return review + refs

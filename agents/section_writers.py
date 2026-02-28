"""
agents/section_writers.py — Focused section writers for the literature review.

Each writer produces one section of the review using a short, focused prompt.
Running sequentially avoids threading issues with Ollama on Windows.
Prompts are capped to keep LLM response time low.
"""

import os
from typing import List, Dict
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from utils.cache import cached_call

load_dotenv()

_OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama2")


def _build_evidence(docs: List[Dict], max_chars: int = 2000) -> str:
    """Formats the source documents into a compact evidence string."""
    evidence = ""
    for i, doc in enumerate(docs, 1):
        source = doc.get("metadata", {}).get("source", "Unknown")
        snippet = doc.get("content", "")[:200].replace("\n", " ")
        evidence += f"[{i}] {snippet}...\n"
        if len(evidence) >= max_chars:
            break
    return evidence


def _make_chain():
    llm = ChatOllama(
        base_url=_OLLAMA_BASE,
        model=_OLLAMA_MODEL,
        temperature=0.3,
        num_predict=350,  # Cap output tokens for speed
    )
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
        "Cite sources using [N] format. Be factual. Max 250 words."
    )

    def write(self, docs: List[Dict], query: str) -> str:
        chain = _make_chain()
        evidence = _build_evidence(docs)
        return cached_call(
            fn=lambda: chain.invoke({
                "instruction": self.INSTRUCTION,
                "query": query,
                "evidence": evidence,
            }),
            key_data={"writer": "methods", "model": _OLLAMA_MODEL,
                      "query": query, "evidence": evidence},
        )


class ResultsSectionWriter:
    """Writes the 'Results & Benchmarks' section of the literature review."""

    INSTRUCTION = (
        "Write a concise 'Results & Benchmarks' section for a literature review. "
        "Summarize empirical results, metrics, and comparisons from the evidence. "
        "Cite sources using [N] format. Be factual. Max 250 words."
    )

    def write(self, docs: List[Dict], query: str) -> str:
        chain = _make_chain()
        evidence = _build_evidence(docs)
        return cached_call(
            fn=lambda: chain.invoke({
                "instruction": self.INSTRUCTION,
                "query": query,
                "evidence": evidence,
            }),
            key_data={"writer": "results", "model": _OLLAMA_MODEL,
                      "query": query, "evidence": evidence},
        )


class ChallengesSectionWriter:
    """Writes the 'Open Challenges & Future Work' section of the literature review."""

    INSTRUCTION = (
        "Write a concise 'Open Challenges & Future Work' section for a literature review. "
        "Identify limitations and open problems from the evidence. "
        "Cite sources using [N] format. Be factual. Max 250 words."
    )

    def write(self, docs: List[Dict], query: str) -> str:
        chain = _make_chain()
        evidence = _build_evidence(docs)
        return cached_call(
            fn=lambda: chain.invoke({
                "instruction": self.INSTRUCTION,
                "query": query,
                "evidence": evidence,
            }),
            key_data={"writer": "challenges", "model": _OLLAMA_MODEL,
                      "query": query, "evidence": evidence},
        )


def merge_sections(methods: str, results: str, challenges: str, query: str) -> str:
    """Combines the three section strings into a single draft review."""
    return (
        f"# Literature Review: {query}\n\n"
        f"## Methods & Approaches\n\n{methods.strip()}\n\n"
        f"## Results & Benchmarks\n\n{results.strip()}\n\n"
        f"## Open Challenges & Future Work\n\n{challenges.strip()}\n"
    )

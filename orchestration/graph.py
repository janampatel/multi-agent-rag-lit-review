
from typing import TypedDict, List, Dict, Optional
from langgraph.graph import StateGraph, END
from agents.query_expansion import QueryExpansionAgent
from agents.screening_council import ScreeningCouncil
from agents.supervisor import SupervisorAgent
from agents.aggregator import AggregatorAgent
from agents.section_writers import (
    MethodsSectionWriter, ResultsSectionWriter,
    ChallengesSectionWriter, merge_sections,
)
from agents.critic import CriticAgent
from rag.retriever import Retriever
from rag.reranker import rerank


# ---------------------------------------------------------------------------
# Rich AgentState
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    # Input
    query: str
    # Retrieval
    expanded_queries: List[str]
    web_search_docs: List[Dict]       # ArXiv docs (optional)
    local_docs: List[Dict]            # Docs from local FAISS store
    merged_docs: List[Dict]           # local + web merged pool
    retrieved_docs: List[Dict]        # Alias kept for backward compat with exporter
    # Loop control
    retrieval_attempts: int
    quality_threshold: int
    # Screening
    screened_docs: List[Dict]
    # Synthesis
    section_methods: str
    section_results: str
    section_challenges: str
    draft_review: str
    # QA
    critic_issues: List[Dict]
    revision_count: int
    # Output
    final_response: str
    exported_files: List[str]


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------

def expand_query(state: AgentState) -> dict:
    agent = QueryExpansionAgent()
    return {"expanded_queries": agent.process(state["query"])}


def retrieve_documents(state: AgentState) -> dict:
    retriever = Retriever()
    all_docs = []
    seen = set()

    # Merge web docs into pool first (they arrive pre-deduplicated)
    for doc in state.get("web_search_docs", []):
        key = doc.get("content", "")[:200]
        if key not in seen:
            all_docs.append(doc)
            seen.add(key)

    for q in state["expanded_queries"]:
        docs = retriever.retrieve(q)
        for doc in docs:
            key = doc.get("content", "")[:200]
            if key not in seen:
                all_docs.append(doc)
                seen.add(key)

    # Cross-encoder rerank the full pool
    reranked = rerank(query=state["query"], docs=all_docs, top_k=10)
    print(f"Retrieved {len(all_docs)} unique docs → reranked to {len(reranked)}")

    return {
        "local_docs": reranked,
        "merged_docs": reranked,
        "retrieved_docs": reranked,
        "retrieval_attempts": state.get("retrieval_attempts", 0) + 1,
    }


def screen_papers(state: AgentState) -> dict:
    council = ScreeningCouncil()
    docs = state.get("merged_docs", [])
    screened = council.vote(docs, state["query"])
    return {
        "screened_docs": screened,
        "retrieved_docs": screened,  # keep exporter compatible
    }


def supervisor_gate(state: AgentState) -> dict:
    # Pure pass-through node — routing logic is in route_supervisor_gate
    threshold = state.get("quality_threshold", 3)
    SupervisorAgent().evaluate_quality(state.get("screened_docs", []), threshold)
    return {}


def write_methods(state: AgentState) -> dict:
    print("[Graph] Writing Methods section...")
    text = MethodsSectionWriter().write(state["screened_docs"], state["query"])
    return {"section_methods": text}


def write_results(state: AgentState) -> dict:
    print("[Graph] Writing Results section...")
    text = ResultsSectionWriter().write(state["screened_docs"], state["query"])
    return {"section_results": text}


def write_challenges(state: AgentState) -> dict:
    print("[Graph] Writing Challenges section...")
    text = ChallengesSectionWriter().write(state["screened_docs"], state["query"])
    return {"section_challenges": text}


def merge_sections_node(state: AgentState) -> dict:
    draft = merge_sections(
        state.get("section_methods", ""),
        state.get("section_results", ""),
        state.get("section_challenges", ""),
        state["query"],
    )
    return {"draft_review": draft}


def critique_review(state: AgentState) -> dict:
    critic = CriticAgent()
    issues = critic.critique(state.get("draft_review", ""), state.get("screened_docs", []))
    return {"critic_issues": issues}


def revise_review(state: AgentState) -> dict:
    """Fallback re-synthesis when the critic finds issues."""
    print("[Graph] Revising review based on critic feedback...")
    issues_text = "; ".join(i["claim"] for i in state.get("critic_issues", []))
    agent = AggregatorAgent()
    original_query = state["query"]
    revised_query  = f"{original_query} [Revision note: fix unsupported citations — {issues_text}]"
    review = agent.synthesize(state.get("screened_docs", []), revised_query)
    return {
        "draft_review": review,
        "final_response": review,
        "revision_count": state.get("revision_count", 0) + 1,
    }


def format_output(state: AgentState) -> dict:
    """Finalise — promote draft_review to final_response."""
    draft = state.get("draft_review", "")
    return {"final_response": draft}


# ---------------------------------------------------------------------------
# Conditional routers
# ---------------------------------------------------------------------------

def route_supervisor_gate(state: AgentState) -> str:
    screened   = state.get("screened_docs", [])
    threshold  = state.get("quality_threshold", 3)
    attempts   = state.get("retrieval_attempts", 0)

    if len(screened) < threshold and attempts < 2:
        print(f"[Gate] Retrying retrieval (attempt {attempts})...")
        return "retry"
    return "synthesize"


def route_after_critique(state: AgentState) -> str:
    issues   = state.get("critic_issues", [])
    revisions = state.get("revision_count", 0)

    if issues and revisions < 1:
        print(f"[Critic] {len(issues)} issue(s) found — revising (round {revisions + 1})...")
        return "revise"
    return "done"


# ---------------------------------------------------------------------------
# Build Graph
# ---------------------------------------------------------------------------
workflow = StateGraph(AgentState)

# Nodes
workflow.add_node("expand_query",     expand_query)
workflow.add_node("retrieve",         retrieve_documents)
workflow.add_node("screen",           screen_papers)
workflow.add_node("supervisor_gate",  supervisor_gate)
workflow.add_node("write_methods",    write_methods)
workflow.add_node("write_results",    write_results)
workflow.add_node("write_challenges", write_challenges)
workflow.add_node("merge_sections",   merge_sections_node)
workflow.add_node("critique_review",  critique_review)
workflow.add_node("revise_review",    revise_review)
workflow.add_node("format_output",    format_output)

# Entry point
workflow.set_entry_point("expand_query")

# Linear edges
workflow.add_edge("expand_query",    "retrieve")
workflow.add_edge("retrieve",        "screen")
workflow.add_edge("screen",          "supervisor_gate")

# Supervisor gate conditional
workflow.add_conditional_edges(
    "supervisor_gate",
    route_supervisor_gate,
    {
        "retry":     "expand_query",   # Broaden queries and retry
        "synthesize": "write_methods", # Proceed to section writers
    },
)

# Sequential section writing → merge
workflow.add_edge("write_methods",    "write_results")
workflow.add_edge("write_results",    "write_challenges")
workflow.add_edge("write_challenges", "merge_sections")
workflow.add_edge("merge_sections",   "critique_review")

# Critic conditional
workflow.add_conditional_edges(
    "critique_review",
    route_after_critique,
    {
        "revise": "revise_review",  # Fix and re-critique
        "done":   "format_output",  # Clean — finalise
    },
)
workflow.add_edge("revise_review", "format_output")
workflow.add_edge("format_output", END)

app = workflow.compile()

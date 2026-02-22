
from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, END
from agents.query_expansion import QueryExpansionAgent
from agents.screening import ScreeningAgent
from agents.aggregator import AggregatorAgent
from rag.retriever import Retriever
from rag.reranker import rerank

# Define State
class AgentState(TypedDict):
    query: str
    expanded_queries: List[str]
    retrieved_docs: List[Dict]
    final_response: str

# Node Functions
def expand_query(state: AgentState):
    agent = QueryExpansionAgent()
    return {"expanded_queries": agent.process(state["query"])}

def retrieve_documents(state: AgentState):
    retriever = Retriever()
    all_docs = []
    seen_contents = set()
    
    for q in state["expanded_queries"]:
        docs = retriever.retrieve(q)
        for doc in docs:
            if doc["content"] not in seen_contents:
                all_docs.append(doc)
                seen_contents.add(doc["content"])
                
    print(f"Total unique documents retrieved: {len(all_docs)}")
    return {"retrieved_docs": all_docs}

def rerank_documents(state: AgentState):
    """
    Cross-encoder re-ranking step.
    Takes the deduplicated pool from retrieve_documents and re-scores
    each (query, chunk) pair using a fine-tuned cross-encoder.
    This runs before the LLM screener, so the screener only sees
    already-high-precision candidates — saving LLM tokens and improving quality.
    """
    reranked = rerank(
        query=state["query"],
        docs=state["retrieved_docs"],
        top_k=10  # Keep top 10 after reranking, screener does final filtering
    )
    return {"retrieved_docs": reranked}

def screen_papers(state: AgentState):
    agent = ScreeningAgent()
    query = state["query"]
    docs = state["retrieved_docs"]
    
    if not docs:
        return {"retrieved_docs": []}
        
    screened_docs = agent.screen(docs, query)
    return {"retrieved_docs": screened_docs}

def synthesize_review(state: AgentState):
    agent = AggregatorAgent()
    review = agent.synthesize(state["retrieved_docs"], state["query"])
    return {"final_response": review}

# Build Graph
workflow = StateGraph(AgentState)

workflow.add_node("expand_query", expand_query)
workflow.add_node("retrieve", retrieve_documents)
workflow.add_node("rerank", rerank_documents)
workflow.add_node("screen", screen_papers)
workflow.add_node("synthesize", synthesize_review)

workflow.set_entry_point("expand_query")
workflow.add_edge("expand_query", "retrieve")
workflow.add_edge("retrieve", "rerank")
workflow.add_edge("rerank", "screen")
workflow.add_edge("screen", "synthesize")
workflow.add_edge("synthesize", END)

app = workflow.compile()


from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from agents.query_expansion import QueryExpansionAgent
from agents.screening import ScreeningAgent
from agents.aggregator import AggregatorAgent
from rag.retriever import Retriever

# Define State
class AgentState(TypedDict):
    query: str
    expanded_queries: List[str]
    retrieved_docs: List[dict]
    final_response: str

# Node Functions
def expand_query(state: AgentState):
    agent = QueryExpansionAgent()
    return {"expanded_queries": agent.process(state["query"])}

def retrieve_documents(state: AgentState):
    retriever = Retriever()
    # Simple loop over queries (MVP just has 1)
    all_docs = []
    for q in state["expanded_queries"]:
        docs = retriever.retrieve(q)
        all_docs.extend(docs)
    return {"retrieved_docs": all_docs}

def synthesize_review(state: AgentState):
    agent = AggregatorAgent()
    review = agent.synthesize(state["retrieved_docs"])
    return {"final_response": review}

# Build Graph
workflow = StateGraph(AgentState)

workflow.add_node("expand_query", expand_query)
workflow.add_node("retrieve", retrieve_documents)
workflow.add_node("synthesize", synthesize_review)

workflow.set_entry_point("expand_query")
workflow.add_edge("expand_query", "retrieve")
workflow.add_edge("retrieve", "synthesize")
workflow.add_edge("synthesize", END)

app = workflow.compile()

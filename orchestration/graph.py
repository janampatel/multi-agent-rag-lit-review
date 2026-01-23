
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
    all_docs = []
    seen_contents = set()
    
    for q in state["expanded_queries"]:
        docs = retriever.retrieve(q)
        for doc in docs:
            # Simple deduplication based on content matching
            # In production, use a unique doc ID or hash
            if doc["content"] not in seen_contents:
                all_docs.append(doc)
                seen_contents.add(doc["content"])
                
    print(f"Total unique documents retrieved: {len(all_docs)}")
    return {"retrieved_docs": all_docs}

def screen_papers(state: AgentState):
    agent = ScreeningAgent()
    # Use the original user query for strict relevance checking
    # Or optimize to check against specific sub-queries if needed
    query = state["query"]
    docs = state["retrieved_docs"]
    
    if not docs:
        return {"retrieved_docs": []}
        
    screened_docs = agent.screen(docs, query)
    return {"retrieved_docs": screened_docs}

def synthesize_review(state: AgentState):
    agent = AggregatorAgent()
    # Pass the query so the LLM knows what to focus on
    review = agent.synthesize(state["retrieved_docs"], state["query"])
    return {"final_response": review}

# Build Graph
workflow = StateGraph(AgentState)

workflow.add_node("expand_query", expand_query)
workflow.add_node("retrieve", retrieve_documents)
workflow.add_node("screen", screen_papers)
workflow.add_node("synthesize", synthesize_review)

workflow.set_entry_point("expand_query")
workflow.add_edge("expand_query", "retrieve")
workflow.add_edge("retrieve", "screen")
workflow.add_edge("screen", "synthesize")
workflow.add_edge("synthesize", END)

app = workflow.compile()


from typing import List, Dict
import os
from dotenv import load_dotenv
from utils.backboard_langchain import BackboardLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

class AggregatorAgent:
    def __init__(self):
        self.llm = BackboardLLM(temperature=0.3, use_memory=False)  # Low temp for factual synthesis
        
        self.prompt = PromptTemplate(
            template="""You are an expert academic researcher. 
            Synthesize a cohesive literature review based ONLY on the provided retrieved text chunks.
            
            Research Topic: {query}
            
            Retrieved Evidence:
            {evidence}
            
            Instructions:
            1. Organize the review logically (e.g., by method, by metrics, etc.).
            2. Cite your sources using the [ID] format inline (e.g., "Method A is efficient [1].").
            3. If the evidence contradicts, mention the contradiction.
            4. Do not invent information not present in the evidence.
            
            Systematic Review:""",
            input_variables=["query", "evidence"]
        )
        self.chain = self.prompt | self.llm | StrOutputParser()

    def synthesize(self, retrieved_data: List[Dict], query: str = "Research Topic") -> str:
        """
        Synthesizes a review from retrieved documents.
        """
        if not retrieved_data:
            return "No relevant documents found to answer the query."
            
        print(f"[Backboard] Synthesizing review from {len(retrieved_data)} documents...")
        
        # Format evidence specifically for the prompt
        evidence_text = ""
        for i, doc in enumerate(retrieved_data):
            cid = i + 1 
            content = doc.get('content', '').strip()
            source = doc.get('metadata', {}).get('source', 'Unknown Source')
            evidence_text += f"[{cid}] (Source: {source}): {content}\n\n"
            
        try:
            review = self.chain.invoke({"query": query, "evidence": evidence_text})
            return review
        except Exception as e:
            print(f"Error in aggregation: {e}")
            return "Failed to synthesize review due to an internal error."

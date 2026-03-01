
from typing import List, Dict
import json
import os
from dotenv import load_dotenv
from utils.backboard_langchain import BackboardLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

class ScreeningAgent:
    def __init__(self):
        self.llm = BackboardLLM(temperature=0, use_memory=False)  # Deterministic for screening
        
        self.prompt = PromptTemplate(
            template="""You are a strict screener for a Systematic Literature Review.
            
            Research Topic: "{query}"
            
            Evaluate the following papers. Return a JSON object with a single key "relevant_ids" containing the list of ID numbers for papers that are RELEVANT to the topic.
            Reject general papers if they don't match the specific research question.
            
            Papers to screen:
            {papers_text}
            
            JSON Output:""",
            input_variables=["query", "papers_text"]
        )
        self.chain = self.prompt | self.llm | JsonOutputParser()

    def screen(self, papers: List[Dict], query: str, batch_size: int = 5) -> List[Dict]:
        """
        Screens a list of papers in batches and returns only relevant ones.
        """
        print(f"[Backboard] Screening {len(papers)} papers for query: '{query}'...")
        relevant_papers = []
        
        # Process in batches
        for i in range(0, len(papers), batch_size):
            batch = papers[i:i+batch_size]
            batch_text = ""
            batch_map = {} 
            
            for idx, p in enumerate(batch):
                simple_id = idx 
                batch_map[simple_id] = p
                content_preview = p.get('content', '')[:300].replace('\n', ' ')
                batch_text += f"[ID {simple_id}]: {content_preview}...\n"
                
            try:
                response = self.chain.invoke({"query": query, "papers_text": batch_text})
                
                # Extract IDs
                kept_ids = response.get("relevant_ids", [])
                
                # Map back to original paper objects
                for kid in kept_ids:
                    try:
                        kid_int = int(kid)
                        if kid_int in batch_map:
                            relevant_papers.append(batch_map[kid_int])
                    except (ValueError, TypeError):
                        continue
                        
            except Exception as e:
                print(f"Error screening batch {i}: {e}")
                # Fallback: keep all if screening fails to avoid data loss
                relevant_papers.extend(batch)
                
        print(f"[Backboard] Screening complete. Kept {len(relevant_papers)}/{len(papers)} papers.")
        return relevant_papers


from typing import List
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import CommaSeparatedListOutputParser
import os
from dotenv import load_dotenv

load_dotenv()

class QueryExpansionAgent:
    def __init__(self):
        self.llm = ChatOllama(
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            model=os.getenv("OLLAMA_MODEL", "llama2"),
            temperature=0.7
        )
        self.output_parser = CommaSeparatedListOutputParser()
        self.prompt = PromptTemplate(
            template="""You are an expert research assistant.
            Generate 3 to 5 diverse search queries based on the user's research topic.
            
            Rules:
            1. Return ONLY the queries separated by commas.
            2. Do NOT number them.
            3. Do NOT say "Here are the queries".
            4. Keep them concise.
            
            Topic: {topic}
            
            Queries:""",
            input_variables=["topic"]
        )
        self.chain = self.prompt | self.llm | self.output_parser

    def process(self, original_query: str) -> List[str]:
        """
        Expands a single research topic into multiple search queries.
        """
        try:
            print(f"Expanding query: '{original_query}'...")
            expanded_queries = self.chain.invoke({"topic": original_query})
            
            # Clean up potentially messy output
            cleaned_queries = [q.strip() for q in expanded_queries if q.strip()]
            
            # Ensure original is kept if model fails or just as a fallback
            if original_query not in cleaned_queries:
                cleaned_queries.insert(0, original_query)
                
            print(f"Generated {len(cleaned_queries)} variations.")
            return cleaned_queries
        except Exception as e:
            print(f"Error in query expansion: {e}")
            return [original_query]

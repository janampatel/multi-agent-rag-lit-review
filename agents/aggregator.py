
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

load_dotenv()

class AggregatorAgent:
    def __init__(self):
        self.llm = ChatOllama(
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            model=os.getenv("OLLAMA_MODEL", "llama3"),
            temperature=0
        )
        self.prompt = PromptTemplate(
            template="""You are an expert research scientist. 
            Synthesize the following retrieved context into a coherent, well-cited answer to the user's research question.
            
            Context:
            {context}
            
            Synthesized Answer:""",
            input_variables=["context"]
        )
        self.chain = self.prompt | self.llm | StrOutputParser()

    def synthesize(self, retrieved_data: list) -> str:
        if not retrieved_data:
            return "No relevant documents found to answer the query."
            
        print(f"Synthesizing answer from {len(retrieved_data)} chunks...")
        
        # Combine content for context
        context_str = "\n\n".join([f"Source: {d.get('metadata', 'Unknown')}\nContent: {d.get('content', '')}" for d in retrieved_data])
        
        try:
            return self.chain.invoke({"context": context_str})
        except Exception as e:
            print(f"Error during synthesis: {e}")
            return "Failed to synthesize response."

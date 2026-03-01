"""
utils/backboard_langchain.py — LangChain-compatible Backboard LLM wrapper.

Allows drop-in replacement of ChatOllama with BackboardLLM in existing agents.
"""

from typing import Any, List, Optional
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from utils.backboard_client import get_backboard_client


class BackboardLLM(LLM):
    """LangChain-compatible wrapper for Backboard API."""
    
    model_name: str = "backboard"
    temperature: float = 0.7
    use_memory: bool = True
    
    @property
    def _llm_type(self) -> str:
        return "backboard"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call Backboard API with the prompt."""
        client = get_backboard_client()
        return client.invoke(prompt, memory=self.use_memory)
    
    @property
    def _identifying_params(self):
        """Return identifying parameters."""
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "use_memory": self.use_memory,
        }

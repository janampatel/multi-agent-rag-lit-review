"""
utils/backboard_client.py — Backboard API client wrapper for Multi-Agent RAG.

Replaces Ollama with Backboard for:
- Persistent memory across runs
- Faster cloud-hosted inference
- Built-in document RAG capabilities
"""

import os
import requests
from typing import Dict, List, Optional
from dotenv import load_dotenv

load_dotenv()


class BackboardClient:
    """Simplified Backboard client for literature review agents."""
    
    def __init__(self):
        self.api_key = os.getenv("BACKBOARD_API_KEY")
        self.assistant_id = os.getenv("BACKBOARD_ASSISTANT_ID")
        self.model_provider = os.getenv("BACKBOARD_MODEL_PROVIDER", "cohere")
        self.model_name = os.getenv("BACKBOARD_MODEL_NAME", "command-r-08-2024")
        self.base_url = "https://app.backboard.io/api"
        self.headers = {"X-API-Key": self.api_key}
        self.thread_id = None
        
        if not self.api_key:
            raise ValueError("BACKBOARD_API_KEY not found in .env")
    
    def _ensure_assistant(self):
        """Create or verify assistant exists."""
        if self.assistant_id:
            # Verify existing assistant
            try:
                response = requests.get(
                    f"{self.base_url}/assistants/{self.assistant_id}",
                    headers=self.headers
                )
                if response.status_code == 200:
                    print(f"[Backboard] Using existing assistant: {self.assistant_id}")
                    return self.assistant_id
            except Exception as e:
                print(f"[Backboard] Assistant verification failed: {e}")
        
        # Create new assistant
        print("[Backboard] Creating new assistant...")
        response = requests.post(
            f"{self.base_url}/assistants",
            headers=self.headers,
            json={
                "name": "Literature Review Assistant",
                "system_prompt": "You are an expert research assistant specializing in systematic literature reviews. Provide concise, factual responses based on evidence.",
                "model_provider": self.model_provider,
                "model_name": self.model_name,
            }
        )
        response.raise_for_status()
        self.assistant_id = response.json()["assistant_id"]
        print(f"[Backboard] Created assistant: {self.assistant_id}")
        return self.assistant_id
    
    def _ensure_thread(self):
        """Create or reuse thread for persistent memory."""
        if self.thread_id:
            return self.thread_id
        
        assistant_id = self._ensure_assistant()
        print("[Backboard] Creating new thread...")
        response = requests.post(
            f"{self.base_url}/assistants/{assistant_id}/threads",
            headers=self.headers,
            json={}
        )
        response.raise_for_status()
        self.thread_id = response.json()["thread_id"]
        print(f"[Backboard] Thread created: {self.thread_id}")
        return self.thread_id
    
    def invoke(self, prompt: str, memory: bool = True) -> str:
        """
        Send a message and get response.
        
        Args:
            prompt: The prompt/question to send
            memory: Whether to use persistent memory (default True)
        
        Returns:
            The assistant's response text
        """
        thread_id = self._ensure_thread()
        
        try:
            response = requests.post(
                f"{self.base_url}/threads/{thread_id}/messages",
                headers=self.headers,
                data={
                    "content": prompt,
                    "memory": "Auto" if memory else "Off",
                    "stream": "false",
                }
            )
            response.raise_for_status()
            result = response.json()
            return result.get("content", "")
        except requests.exceptions.HTTPError as e:
            error_msg = f"Backboard API Error: {e.response.status_code} - {e.response.text}"
            print(f"[Backboard] {error_msg}")
            raise Exception(error_msg)
    
    def upload_document(self, file_path: str) -> str:
        """
        Upload a PDF to the assistant for RAG.
        
        Args:
            file_path: Path to the PDF file
        
        Returns:
            Document ID
        """
        assistant_id = self._ensure_assistant()
        
        with open(file_path, "rb") as f:
            files = {"file": (os.path.basename(file_path), f, "application/pdf")}
            response = requests.post(
                f"{self.base_url}/assistants/{assistant_id}/documents",
                headers=self.headers,
                files=files
            )
        
        response.raise_for_status()
        doc_id = response.json().get("document_id")
        print(f"[Backboard] Uploaded document: {os.path.basename(file_path)} -> {doc_id}")
        return doc_id
    
    def add_memory(self, key: str, value: str) -> str:
        """
        Manually add a memory fact.
        
        Args:
            key: Memory key/label
            value: Memory content
        
        Returns:
            Memory ID
        """
        assistant_id = self._ensure_assistant()
        
        response = requests.post(
            f"{self.base_url}/assistants/{assistant_id}/memories",
            headers=self.headers,
            json={"key": key, "value": value}
        )
        response.raise_for_status()
        return response.json().get("memory_id")
    
    def reset_thread(self):
        """Start a new conversation thread."""
        self.thread_id = None
        print("[Backboard] Thread reset - next call will create new thread")


# Singleton instance
_client: Optional[BackboardClient] = None


def get_backboard_client() -> BackboardClient:
    """Get or create the global Backboard client instance."""
    global _client
    if _client is None:
        _client = BackboardClient()
    return _client

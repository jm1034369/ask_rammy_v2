import os
from typing import Dict, List, Optional

import requests

class LocalGenerator:
    """Generate chatbot answers using an Ollama local model."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> None:
        self.base_url = (
            base_url
            or os.getenv("LOCAL_MODEL_BASE_URL", "http://ollama:11434")
        ).rstrip("/")

        self.model = model or os.getenv("LOCAL_GENERATOR_MODEL", "llama3.2:3b")

        self.timeout = (
            timeout
            if timeout is not None
            else float(os.getenv("LOCAL_GENERATOR_TIMEOUT", "20"))
        )

    def generate(self, messages: List[Dict[str, str]]) -> str:
        response = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "num_predict": 300,
                },
            },
            timeout=self.timeout,
        )

        response.raise_for_status()

        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("Ollama returned an invalid response.")

        message = payload.get("message")
        if not isinstance(message, dict):
            raise ValueError("Ollama response is missing a message.")

        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Ollama returned an empty or invalid answer")

        return content.strip()
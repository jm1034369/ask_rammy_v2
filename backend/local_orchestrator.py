"""Optional local query-orchestration layer for the RAG backend.

The local model is used for query planning only. Answer generation remains
owned by the configured answer model, so disabling this layer preserves the
existing chatbot behavior.
"""

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class OrchestrationDecision:
    """The retrieval query and the reason it was selected."""

    query: str
    source: str


class LocalOrchestrator:
    """Use an Ollama-compatible local model to plan a RAG query.

    The network call is deliberately optional and bounded. Any local model
    failure returns the user's original text so the cloud-backed RAG path can
    continue serving requests.
    """

    def __init__(
        self,
        enabled: Optional[bool] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> None:
        configured = os.getenv("LOCAL_ORCHESTRATOR_ENABLED", "false").lower()
        self.enabled = enabled if enabled is not None else configured in {"1", "true", "yes", "on"}
        self.base_url = (base_url or os.getenv("LOCAL_MODEL_BASE_URL", "http://host.docker.internal:11434")).rstrip("/")
        self.model = model or os.getenv("LOCAL_MODEL", "llama3.2:3b")
        self.timeout = timeout if timeout is not None else float(os.getenv("LOCAL_MODEL_TIMEOUT", "4"))

    def decide(self, question: str, history: Optional[List[Dict[str, Any]]] = None) -> OrchestrationDecision:
        """Return a standalone retrieval query without blocking the fallback path."""
        original = (question or "").strip()
        if not original or not self.enabled:
            return OrchestrationDecision(original, "original")

        prompt = self._planning_prompt(original, history or [])
        try:
            import requests
        except ImportError:
            print("[orchestrator] requests is unavailable; using the original query.")
            return OrchestrationDecision(original, "fallback")

        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "stream": False,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "Rewrite the user's message as one concise standalone HR knowledge-base "
                                "search query. Output only the query, with no quotes or explanation."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    "options": {"temperature": 0},
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            content = response.json().get("message", {}).get("content", "")
            planned = " ".join(str(content).split()).strip(" \"'")
            if planned and len(planned) <= 500:
                return OrchestrationDecision(planned, "local")
        except (requests.RequestException, ValueError, TypeError, AttributeError) as error:
            print(f"[orchestrator] Local planner unavailable: {error}")

        return OrchestrationDecision(original, "fallback")

    @staticmethod
    def _planning_prompt(question: str, history: List[Dict[str, Any]]) -> str:
        recent = [
            f"{item.get('role', 'user')}: {item.get('content', '')}"
            for item in history[-4:]
            if isinstance(item, dict) and item.get("content")
        ]
        context = "\n".join(recent)
        return f"Conversation:\n{context}\n\nCurrent message:\n{question}" if context else question
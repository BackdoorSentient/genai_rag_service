# app/llm/ollama_client.py
import httpx
from app.llm.base import BaseLLM

class OllamaClient(BaseLLM):
    def __init__(self, model: str):
        self.model = model
        self.base_url = "http://localhost:11434"

    async def generate(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }
        async with httpx.AsyncClient(timeout=60) as client:
            res = await client.post(f"{self.base_url}/api/generate", json=payload)
            res.raise_for_status()
            return res.json().get("response", "").strip()

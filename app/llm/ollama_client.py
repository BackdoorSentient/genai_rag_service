# app/llm/ollama_client.py
import httpx
from app.llm.base import BaseLLM

class OllamaClient(BaseLLM):
    def __init__(self, model: str = "llama3"):
        self.model = model
        # self.base_url = "http://localhost:11434"
        self.base_url = "http://127.0.0.1:11434"

    async def generate(self, prompt: str) -> str:
        print("MODEL:", self.model, type(self.model))
        print("BASE_URL:", self.base_url, type(self.base_url))
        print("PROMPT TYPE:", type(prompt))
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }

        # async with httpx.AsyncClient(timeout=httpx.Timeout(120.0),trust_env=False ) as client:
        async with httpx.AsyncClient(timeout=120.0,trust_env=False) as client:
            res = await client.post(
                f"{self.base_url}/api/generate",
                json=payload
            )

            if res.status_code != 200:
                raise Exception(f"Ollama error: {res.text}")

            data = res.json()
            return (data.get("response") or "").strip()

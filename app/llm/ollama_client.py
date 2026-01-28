import asyncio
from app.llm.base import BaseLLM
from app.utils.retry import retry_async
from app.utils.circuit_breaker import CircuitBreaker
from config.settings import settings


class OllamaClient(BaseLLM):
    def __init__(self):
        self.model = settings.ollama_model
        self.breaker = CircuitBreaker()

    async def _call_llm(self, prompt: str) -> str:
        process = await asyncio.create_subprocess_exec(
            "ollama", "run", self.model,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, _ = await process.communicate(prompt.encode())
        return stdout.decode().strip()

    async def generate(self, prompt: str) -> str:
        if not self.breaker.allow_request():
            raise RuntimeError("LLM circuit breaker open")

        try:
            result = await retry_async(
                lambda: self._call_llm(prompt),
                retries=3,
                timeout=20
            )
            self.breaker.record_success()
            return result
        except Exception as e:
            self.breaker.record_failure()
            raise e

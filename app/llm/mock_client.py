from app.llm.base import BaseLLM
import asyncio

class MockLLM(BaseLLM):
    async def generate(self, prompt: str) -> str:
        await asyncio.sleep(0.1)
        return "This is a mocked LLM response for OpenAI/Azure."

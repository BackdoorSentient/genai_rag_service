# app/llm/openai_client.py
import openai
from app.llm.base import BaseLLM
from config.settings import settings

class OpenAIClient(BaseLLM):
    def __init__(self):
        openai.api_key = settings.openai_api_key
        self.model = settings.openai_model

    async def generate(self, prompt: str) -> str:
        response = await openai.ChatCompletion.acreate(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()

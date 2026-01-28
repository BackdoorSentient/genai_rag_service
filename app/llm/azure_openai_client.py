# app/llm/azure_openai_client.py
from openai import OpenAI
from app.llm.base import BaseLLM
from config.settings import settings

class AzureOpenAIClient(BaseLLM):
    def __init__(self):
        self.client = OpenAI(
            api_key=settings.azure_openai_key,
            base_url=settings.azure_openai_endpoint,
            organization=None,
        )
        self.deployment = settings.azure_openai_deployment
        self.model = settings.azure_openai_version

    async def generate(self, prompt: str) -> str:
        response = await self.client.chat.completions.create(
            deployment_id=self.deployment,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content.strip()

# app/llm/factory.py
from config.settings import settings
from app.llm.ollama_client import OllamaClient
from app.llm.openai_client import OpenAIClient
from app.llm.azure_openai_client import AzureOpenAIClient
from app.llm.mock_client import MockLLM

def get_llm():
    provider = settings.llm_provider.lower()

    if provider == "openai":
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAI")
        return OpenAIClient()

    elif provider == "azure":
        if not all([
            settings.azure_openai_key,
            settings.azure_openai_endpoint,
            settings.azure_openai_deployment,
            settings.azure_openai_version
        ]):
            raise ValueError("Azure OpenAI config incomplete")
        return AzureOpenAIClient()

    elif provider == "mock":
        return MockLLM()

    else:  # "ollama"
        return OllamaClient(settings.ollama_model)

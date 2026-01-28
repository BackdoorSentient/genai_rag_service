from config.settings import settings
from app.llm.ollama_client import OllamaClient
from app.llm.mock_client import MockLLM

def get_llm():
    provider = settings.llm_provider.lower()
    if provider == "openai":
        return MockLLM()
    elif provider == "azure":
        return MockLLM()
    else:
        return OllamaClient()

from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent 
# ENV_PATH = BASE_DIR / ".env"
# BASE_DIR = Path("C:/Users/ANIKET/OneDrive/Desktop/gen_ai_rag_services")
ENV_PATH = BASE_DIR / ".env"

class Settings(BaseSettings):
    """
    Centralized application configuration.
    Values are loaded from environment variables and .env file.
    """

    model_config = SettingsConfigDict(
        # env_file=".env",
        env_file=str(ENV_PATH),
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
    )
#     model_config = SettingsConfigDict(
#     env_file_encoding="utf-8",
#     env_ignore_empty=True,
#     extra="ignore",
# )

    #llm config
    llm_provider: str = "ollama"  # ollama,openai,azure
    ollama_model: str = "llama3"

    openai_api_key: Optional[str] = None
    openai_model: Optional[str] = None

    azure_openai_key: Optional[str] = None
    azure_openai_endpoint: Optional[str] = None
    azure_openai_deployment: Optional[str] = None
    azure_openai_version: Optional[str] = None

    #embeddings/vector Store
    data_dir: str = "data/raw"

    #required keys#
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    faiss_index_path: str = "data/processed/faiss_index"

    pinecone_api_key: Optional[str] = None
    pinecone_environment: Optional[str] = None
    pinecone_index_name: Optional[str] = None
    use_pinecone: Optional[bool] = False

    #server for fastapi
    host: str = "127.0.0.1"
    port: int = 8000

    retry_count: int = 3
    timeout_seconds: int = 20
    circuit_breaker_threshold: int = 3
    circuit_breaker_recovery: int = 30

settings = Settings()

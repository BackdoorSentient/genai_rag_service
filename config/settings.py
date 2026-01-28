from pydantic import BaseSettings


class Settings(BaseSettings):
    data_dir: str
    embedding_model: str
    ollama_model: str
    faiss_index_path: str

    class Config:
        env_file = ".env"


settings = Settings()

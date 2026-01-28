import os
from typing import List, Dict
from langchain.vectorstores import FAISS, Pinecone
from langchain.embeddings import HuggingFaceEmbeddings
from config.settings import settings

# Mock Pinecone if USE_PINECONE=False or no API key
USE_PINECONE = getattr(settings, "USE_PINECONE", False)

class VectorStore:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)
        self.db = None

    def build_or_load(self, documents: List[Dict]):
        if USE_PINECONE:
            try:
                import pinecone
                pinecone.init(
                    api_key=settings.PINECONE_API_KEY,
                    environment=settings.PINECONE_ENVIRONMENT
                )
                index_name = settings.PINECONE_INDEX_NAME
                if index_name not in pinecone.list_indexes():
                    pinecone.create_index(index_name, dimension=self.embeddings.embed_query("test").shape[0])
                self.db = Pinecone.from_texts(
                    [d["text"] for d in documents],
                    embedding=self.embeddings,
                    index_name=index_name
                )
                return
            except Exception:
                print("Pinecone not available. Using local FAISS.")

        # Fallback: FAISS
        if os.path.exists(settings.faiss_index_path):
            self.db = FAISS.load_local(settings.faiss_index_path, self.embeddings)
        else:
            texts = [d["text"] for d in documents]
            metadatas = [d["metadata"] for d in documents]
            self.db = FAISS.from_texts(texts=texts, embedding=self.embeddings, metadatas=metadatas)
            self.db.save_local(settings.faiss_index_path)

    def search(self, query: str, k: int = 4):
        return self.db.similarity_search(query, k=k)

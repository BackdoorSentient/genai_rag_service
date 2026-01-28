import os
from typing import List, Dict
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from config.settings import settings


class VectorStore:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model
        )
        self.db = None

    def build_or_load(self, documents: List[Dict]):
        if os.path.exists(settings.faiss_index_path):
            self.db = FAISS.load_local(
                settings.faiss_index_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
        else:
            texts = [d["text"] for d in documents]
            metadatas = [d["metadata"] for d in documents]

            self.db = FAISS.from_texts(
                texts=texts,
                embedding=self.embeddings,
                metadatas=metadatas
            )
            self.db.save_local(settings.faiss_index_path)

    def search(self, query: str, k: int = 4):
        return self.db.similarity_search(query, k=k)

from typing import List, Dict
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings


class VectorStore:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.db = None

    def build(self, documents: List[Dict]):
        texts = [d["text"] for d in documents]
        metadatas = [d["metadata"] for d in documents]

        self.db = FAISS.from_texts(
            texts=texts,
            embedding=self.embeddings,
            metadatas=metadatas
        )

    def search(self, query: str, k: int = 4):
        return self.db.similarity_search(query, k=k)

import os
from typing import List, Dict

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS, Pinecone

from config.settings import settings


class VectorStore:
    def __init__(self):
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model
        )
        self.db = None

    def build_or_load(self, documents: List[Dict]):
        """
        Build or load vector store using Pinecone or FAISS.
        """

        # ----------------------------
        # Pinecone (optional)
        # ----------------------------
        if settings.use_pinecone:
            try:
                import pinecone

                pinecone.init(
                    api_key=settings.pinecone_api_key,
                    environment=settings.pinecone_environment,
                )

                index_name = settings.pinecone_index_name

                if index_name not in pinecone.list_indexes():
                    pinecone.create_index(
                        name=index_name,
                        dimension=len(self.embeddings.embed_query("test")),
                    )

                self.db = Pinecone.from_texts(
                    texts=[d["text"] for d in documents if d.get("text")],
                    embedding=self.embeddings,
                    index_name=index_name,
                )
                return

            except Exception as e:
                print(f"[WARN] Pinecone unavailable, falling back to FAISS: {e}")

        # ----------------------------
        # FAISS (default)
        # ----------------------------
        texts = [d["text"] for d in documents if d.get("text")]
        metadatas = [d.get("metadata", {}) for d in documents if d.get("text")]

        if not texts:
            raise ValueError(
                "No documents found to build FAISS index. "
                "Ensure documents contain non-empty 'text'."
            )

        # ✅ CRITICAL FIX: check actual FAISS file
        faiss_file = os.path.join(settings.faiss_index_path, "index.faiss")

        if os.path.exists(faiss_file):
            # Load existing FAISS index
            self.db = FAISS.load_local(
                settings.faiss_index_path,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
        else:
            # Build new FAISS index
            self.db = FAISS.from_texts(
                texts=texts,
                embedding=self.embeddings,
                metadatas=metadatas,
            )

            os.makedirs(settings.faiss_index_path, exist_ok=True)
            self.db.save_local(settings.faiss_index_path)

    def search(self, query: str, k: int = 4):
        """
        Perform similarity search.
        """
        if not self.db:
            raise RuntimeError("Vector store not initialized. Call build_or_load() first.")

        return self.db.similarity_search(query, k=k)

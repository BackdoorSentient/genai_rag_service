from app.llm.ollama_client import OllamaClient


class RAGPipeline:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.llm = OllamaClient()

    def ask(self, question: str) -> str:
        docs = self.vector_store.search(question)

        context = "\n\n".join([d.page_content for d in docs])

        prompt = f"""
        Answer the question using the context below.
        If the answer is not present, say you don't know.

        Context:
        {context}

        Question:
        {question}
        """

        return self.llm.generate(prompt)

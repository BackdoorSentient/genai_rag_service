from app.llm.ollama_client import OllamaClient


class RAGPipeline:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.llm = OllamaClient()

    async def ask(self, question: str):
        docs = self.vector_store.search(question)

        context = "\n\n".join(d.page_content for d in docs)
        sources = list({d.metadata.get("source") for d in docs})

        prompt = f"""
Answer ONLY using the context.
If unsure, say "I don't know".

Context:
{context}

Question:
{question}
"""

        try:
            answer = await self.llm.generate(prompt)
            return {"answer": answer, "sources": sources}
        except Exception as e:
            return {
                "answer": "LLM temporarily unavailable. Please try again.",
                "error": str(e),
            }

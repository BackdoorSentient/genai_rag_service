from app.llm.factory import get_llm

class RAGPipeline:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.llm = get_llm()

    async def ask(self, question: str):
        docs = self.vector_store.search(question)

        context = "\n\n".join(d.page_content for d in docs)
        sources = list({d.metadata.get("source") for d in docs})

        prompt = f"""
Answer ONLY using the context below.
If unsure, say "I don't know".

Context:
{context}

Question:
{question}
"""
        try:
            answer = await self.llm.generate(prompt)
            return {"answer": answer, "sources": sources}
        except Exception:
            return {"answer": "LLM temporarily unavailable. Please try again."}

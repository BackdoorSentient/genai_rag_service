from app.llm.ollama_client import OllamaClient
class RAGPipeline:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.llm = OllamaClient()

    async def ask(self, question: str) -> dict:
        docs = self.vector_store.search(question)

        context_blocks = []
        sources = set()

        for d in docs:
            context_blocks.append(d.page_content)
            sources.add(d.metadata.get("source"))

        context = "\n\n".join(context_blocks)

        prompt = f"""
You are a helpful assistant.
Answer the question ONLY using the context below.
If not found, say "I don't know".

Context:
{context}

Question:
{question}
"""

        answer = await self.llm.generate(prompt)

        return {
            "answer": answer,
            "sources": list(sources)
        }

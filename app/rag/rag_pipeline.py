from typing import Dict, Any, List
from app.llm.factory import get_llm


class RAGPipeline:
    """
    Retrieval-Augmented Generation pipeline.
    - Retrieves relevant documents from vector store
    - Builds grounded prompt
    - Calls LLM
    """

    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.llm = get_llm()

    async def ask(self, question: str) -> Dict[str, Any]:
        """
        Answer a question using retrieved context.
        """

        # 1. Retrieve relevant documents
        docs = self.vector_store.search(question)

        if not docs:
            return {
                "answer": "No relevant context found.",
                "sources": []
            }

        # 2. Build context
        context = "\n\n".join(d.page_content for d in docs)
        sources = list({d.metadata.get("source") for d in docs if d.metadata})

        # 3. Prompt (strict grounding)
        prompt = f"""
You are a helpful assistant.
Answer ONLY using the context below.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{question}
""".strip()

        # 4. Call LLM
        try:
            answer = await self.llm.generate(prompt)
            print(f"answer:{answer}")
            print(f"sources:{sources}")
            return {
                "answer": answer,
                "sources": sources
            }

        except Exception:
            return {
                "answer": "LLM temporarily unavailable. Please try again later.",
                "sources": sources
            }

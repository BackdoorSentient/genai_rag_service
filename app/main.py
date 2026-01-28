from fastapi import FastAPI
from app.rag.ingest import load_and_chunk_docs
from app.rag.vector_store import VectorStore
from app.rag.rag_pipeline import RAGPipeline
from config.settings import settings

app = FastAPI()

vector_store = VectorStore()
rag_pipeline = None


@app.on_event("startup")
def startup():
    global rag_pipeline
    docs = load_and_chunk_docs(settings.data_dir)
    vector_store.build_or_load(docs)
    rag_pipeline = RAGPipeline(vector_store)


@app.get("/ask")
async def ask(question: str):
    return await rag_pipeline.ask(question)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", reload=True)

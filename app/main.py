from fastapi import FastAPI
from app.rag.ingest import load_and_chunk_docs
from app.rag.vector_store import VectorStore
from app.rag.rag_pipeline import RAGPipeline

app = FastAPI()

vector_store = VectorStore()
rag_pipeline = None


@app.on_event("startup")
def startup():
    global rag_pipeline
    docs = load_and_chunk_docs("data/raw")
    vector_store.build(docs)
    rag_pipeline = RAGPipeline(vector_store)


@app.get("/ask")
def ask(question: str):
    return {"answer": rag_pipeline.ask(question)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )

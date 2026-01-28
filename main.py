from fastapi import FastAPI, HTTPException
from app.rag.ingest import load_and_chunk_docs
from app.rag.vector_store import VectorStore
from app.rag.rag_pipeline import RAGPipeline
from config.settings import settings

app = FastAPI()

vector_store = VectorStore()
rag_pipeline: RAGPipeline | None = None


@app.post("/load_document")
async def load_document():
    global rag_pipeline

    docs = load_and_chunk_docs(settings.data_dir)

    if not docs:
        raise HTTPException(
            status_code=400,
            detail="No documents found in data directory"
        )

    vector_store.build_or_load(docs)
    rag_pipeline = RAGPipeline(vector_store)

    return {
        "status": "success",
        "documents_loaded": len(docs)
    }


@app.get("/ask")
async def ask(question: str):
    if rag_pipeline is None:
        raise HTTPException(
            status_code=400,
            detail="Documents not loaded. Call /load_document first."
        )

    return await rag_pipeline.ask(question)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000)

from fastapi import FastAPI, HTTPException
from app.rag.ingest import load_and_chunk_docs
from app.rag.vector_store import VectorStore
from app.rag.rag_pipeline import RAGPipeline
from config.settings import settings

app = FastAPI(title="GenAI RAG Service")

# Global singletons
vector_store = VectorStore()
rag_pipeline: RAGPipeline | None = None


# ----------------------------
# Startup: auto-load FAISS + RAG
# ----------------------------
@app.on_event("startup")
async def startup_event():
    global rag_pipeline
    try:
        docs = load_and_chunk_docs(settings.data_dir)
        vector_store.build_or_load(docs)
        rag_pipeline = RAGPipeline(vector_store)
        print("RAG pipeline initialized on startup")
    except Exception as e:
        print(f"Startup failed: {e}")
        rag_pipeline = None


# ----------------------------
# Reload documents (manual)
# ----------------------------
@app.post("/reload")
async def reload_documents():
    global rag_pipeline
    try:
        docs = load_and_chunk_docs(settings.data_dir)
        vector_store.build_or_load(docs)
        rag_pipeline = RAGPipeline(vector_store)
        return {"status": "Index rebuilt successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ----------------------------
# Ask a question
# ----------------------------
@app.get("/ask")
async def ask(question: str):
    if not rag_pipeline:
        raise HTTPException(
            status_code=503,
            detail="RAG pipeline not initialized. Try /reload.",
        )

    return await rag_pipeline.ask(question)


# ----------------------------
# Health check
# ----------------------------
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "rag_ready": rag_pipeline is not None,
        "vector_store_loaded": vector_store.db is not None,
    }


# ----------------------------
# Local run
# ----------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
    )

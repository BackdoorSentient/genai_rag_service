from fastapi import FastAPI, HTTPException
from app.rag.ingest import load_and_chunk_docs
from app.rag.vector_store import VectorStore
from app.rag.rag_pipeline import RAGPipeline
from config.settings import settings

import subprocess
import time
import requests
import atexit

app = FastAPI(title="GenAI RAG Service")

vector_store = VectorStore()
rag_pipeline: RAGPipeline | None = None
ollama_proc: subprocess.Popen | None = None

OLLAMA_URL = "http://127.0.0.1:11434"


def start_ollama_server():
    """Start Ollama in the background and wait until it is ready."""
    global ollama_proc
    print("Starting Ollama server...")
    ollama_proc = subprocess.Popen(["ollama", "serve"])
    
    start_time = time.time()
    timeout = 30
    while True:
        try:
            res = requests.get(OLLAMA_URL)
            if res.status_code == 200:
                print("Ollama server is ready!")
                break
        except requests.ConnectionError:
            pass

        if time.time() - start_time > timeout:
            print("Timeout waiting for Ollama server")
            ollama_proc.terminate()
            raise RuntimeError("Ollama server did not start in time")
        time.sleep(1)


def stop_ollama_server():
    """Terminate Ollama when the app stops."""
    global ollama_proc
    if ollama_proc:
        print("Shutting down Ollama server...")
        ollama_proc.terminate()
        ollama_proc = None

# Ensure Ollama is terminated on process exit
atexit.register(stop_ollama_server)

# auto-load FAISS + RAG
@app.on_event("startup")
async def startup_event():
    global rag_pipeline
    try:
        start_ollama_server() 
        docs = load_and_chunk_docs(settings.data_dir)
        vector_store.build_or_load(docs)
        rag_pipeline = RAGPipeline(vector_store)
        print("RAG pipeline initialized on startup")
    except Exception as e:
        print(f"Startup failed: {e}")
        rag_pipeline = None

# Reload documents
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

@app.get("/ask")
async def ask(question: str):
    if not rag_pipeline:
        raise HTTPException(
            status_code=503,
            detail="RAG pipeline not initialized. Try /reload.",
        )

    return await rag_pipeline.ask(question)

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "rag_ready": rag_pipeline is not None,
        "vector_store_loaded": vector_store.db is not None,
        "ollama_running": ollama_proc is not None
    }

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
    )

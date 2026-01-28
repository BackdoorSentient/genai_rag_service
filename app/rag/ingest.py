from pathlib import Path
from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader

def load_and_chunk_docs(data_dir: str) -> List[Dict]:
    documents = []
    pdf_files = Path(data_dir).glob("*.pdf")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    for pdf in pdf_files:
        loader = PyPDFLoader(str(pdf))
        pages = loader.load()

        chunks = splitter.split_documents(pages)

        for chunk in chunks:
            documents.append({
                "text": chunk.page_content,
                "metadata": {
                    "source": pdf.name,
                    "page": chunk.metadata.get("page", None)
                }
            })

    if not documents:
        raise ValueError(f"No PDFs found or PDFs are empty in {data_dir}")

    return documents

# from pathlib import Path
# from typing import List, Dict
# from langchain.text_splitter import RecursiveCharacterTextSplitter

# def load_and_chunk_docs(data_dir: str) -> List[Dict]:
#     """
#     Load all .txt files from data_dir and split into chunks.
#     Returns a list of dicts with 'text' and 'metadata'.
#     """
#     documents = []
#     txt_files = Path(data_dir).glob("*.txt")

#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=500,
#         chunk_overlap=100
#     )

#     for txt_file in txt_files:
#         content = txt_file.read_text(encoding="utf-8")

#         # Split into chunks
#         chunks = splitter.split_text(content)

#         for i, chunk_text in enumerate(chunks):
#             documents.append({
#                 "text": chunk_text,
#                 "metadata": {
#                     "source": txt_file.name,
#                     "chunk": i
#                 }
#             })

#     return documents


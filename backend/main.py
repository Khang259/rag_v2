from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import os

from backend.utils import load_documents_from_folder, split_documents
from backend.vector_store import VectorStoreManager
from backend.rag import RAGChain

app = FastAPI(title="RAG Backend")

vector_manager = VectorStoreManager()
rag_chain = None  # Lazy init

class IndexRequest(BaseModel):
    folders: List[str]


class QueryRequest(BaseModel):
    question: str
    chat_history: List[dict]  # [{"role": "user/assistant", "content": str}]


@app.post("/index_documents")
def index_documents(request: IndexRequest):
    global rag_chain
    try:
        all_docs = []
        for folder in request.folders:
            if not os.path.exists(folder):
                raise ValueError(f"Folder không tồn tại: {folder}")
            
            docs_in_folder = load_documents_from_folder(folder)
            if docs_in_folder:
                all_docs.extend(docs_in_folder)
            else:
                print(f"Warning: Folder {folder} không chứa file PDF/TXT/DOCX hợp lệ")

        if not all_docs:
            raise ValueError("Không tìm thấy tài liệu hợp lệ.")

        chunks = split_documents(all_docs)
        vector_manager.create_or_update(chunks)
        rag_chain = RAGChain(vector_manager.load())  # Init chain sau index
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/query")
def query_rag(request: QueryRequest):
    global rag_chain
    if rag_chain is None:
        vector_manager.load()
        rag_chain = RAGChain(vector_manager.load())

    try:
        # Convert chat_history sang langchain messages
        from langchain_core.messages import AIMessage, HumanMessage
        history = []
        for msg in request.chat_history:
            if msg["role"] == "user":
                history.append(HumanMessage(content=msg["content"]))
            else:
                history.append(AIMessage(content=msg["content"]))

        response = rag_chain.invoke({
            "question": request.question,
            "chat_history": history
        })
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
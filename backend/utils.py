# backend/utils.py

from pathlib import Path
from typing import List
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import tiktoken

# ── Tokenizer cho mxbai-embed-large ──
EMBEDDING_TOKENIZER = tiktoken.get_encoding("cl100k_base")
MAX_EMBEDDING_TOKENS = 512
RECOMMENDED_TOKENS = 380

def count_tokens(text: str) -> int:
    return len(EMBEDDING_TOKENIZER.encode(text))

def load_documents_from_folder(folder_path: str) -> List[Document]:
    """Load recursive tất cả file .pdf, .txt, .md, .docx trong folder và subfolders"""
    docs = []
    folder = Path(folder_path)
    
    if not folder.exists() or not folder.is_dir():
        return []
    
    for file_path in folder.rglob("*"):
        if not file_path.is_file():
            continue
            
        ext = file_path.suffix.lower()
        try:
            if ext == ".pdf":
                loader = PyPDFLoader(str(file_path))
            elif ext in [".txt", ".md"]:
                loader = TextLoader(str(file_path), encoding="utf-8")
            elif ext in [".docx", ".doc"]:
                loader = Docx2txtLoader(str(file_path))
            else:
                continue
                
            loaded = loader.load()
            for doc in loaded:
                doc.metadata["source_file"] = str(file_path.relative_to(folder))
                doc.metadata["full_path"] = str(file_path)
            docs.extend(loaded)
            
        except Exception as e:
            print(f"Không thể load file {file_path}: {str(e)}")
            continue
    
    return docs

def split_if_too_long(docs: List[Document]) -> List[Document]:
    final_chunks = []
    for doc in docs:
        token_count = count_tokens(doc.page_content)
        if token_count <= MAX_EMBEDDING_TOKENS:
            final_chunks.append(doc)
            continue
            
        print(f"Chunk dài ({token_count} tokens) → cắt nhỏ: {doc.metadata.get('source_file', 'unknown')}")
        small_splitter = RecursiveCharacterTextSplitter(
            chunk_size=RECOMMENDED_TOKENS * 4,
            chunk_overlap=100,
            length_function=count_tokens,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        sub_chunks = small_splitter.split_documents([doc])
        final_chunks.extend(sub_chunks)
    return final_chunks

def split_documents(documents: List[Document]) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150,
        length_function=len,
    )
    
    initial_chunks = text_splitter.split_documents(documents)
    safe_chunks = split_if_too_long(initial_chunks)
    
    print(f"Chunk ban đầu: {len(initial_chunks)} → Sau kiểm tra token: {len(safe_chunks)}")
    return safe_chunks
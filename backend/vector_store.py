from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from typing import List

EMBEDDING_MODEL = "mxbai-embed-large:latest"
PERSIST_DIRECTORY = "../chroma_db_user_docs"  # Relative từ backend


class VectorStoreManager:
    """Class quản lý vector store, tuân thủ Single Responsibility."""
    def __init__(self):
        self.embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
        self.vectorstore = None

    def create_or_update(self, chunks: List):
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=PERSIST_DIRECTORY,
            collection_name="user_documents"
        )

    def load(self):
        if self.vectorstore is None:
            self.vectorstore = Chroma(
                persist_directory=PERSIST_DIRECTORY,
                embedding_function=self.embeddings,
                collection_name="user_documents"
            )
        return self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
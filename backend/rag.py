from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

LLM_MODEL = "qwen2.5-coder:3b"


class RAGChain:
    """Class setup RAG chain, dễ inject dependencies."""
    def __init__(self, retriever):
        self.llm = ChatOllama(
            model=LLM_MODEL,
            temperature=0.3,
            num_ctx=8192,
        )
        self.prompt = self._create_prompt()
        self.chain = self._create_chain(retriever)

    def _create_prompt(self):
        system_prompt = """Bạn là trợ lý AI chuyên nghiệp, trung thực và chính xác.
Bạn chỉ được trả lời dựa trên thông tin trong các tài liệu được cung cấp dưới đây.
Nếu câu hỏi không liên quan đến tài liệu hoặc bạn không tìm thấy thông tin → hãy trả lời trung thực rằng bạn không biết hoặc thông tin không có trong tài liệu.

Nguyên tắc trả lời:
• Trả lời bằng tiếng Việt tự nhiên, rõ ràng, mạch lạc
• Sử dụng markdown khi cần (danh sách, code block, in đậm,...)
• Trình bày ngắn gọn nhưng đầy đủ ý
• Không bịa đặt, không suy diễn quá mức

Context (tài liệu tham khảo):
{context}

Câu hỏi của người dùng: """
        return ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])

    def _create_chain(self, retriever):
        def format_docs(docs):
            return "\n\n".join([
                f"[Trang {d.metadata.get('page', 'N/A')}] {d.page_content}"
                for d in docs
            ])

        return (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough(),
                "chat_history": RunnablePassthrough()
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def invoke(self, input_data: dict):
        return self.chain.invoke(input_data)
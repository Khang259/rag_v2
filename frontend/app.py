import streamlit as st
import requests
import os
import json

# Config
API_URL = "http://127.0.0.1:8000"  # FastAPI server
VECTOR_DB_PATH = "./chroma_db_user_docs"

# Custom CSS với color scheme và typography (Be Vietnam Pro)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;600;700&display=swap');
    
    :root {
        --primary-blue: #3b82f6; /* blue-500 */
        --hover-blue: #2563eb;   /* blue-600 */
        --light-blue: #eff6ff;   /* blue-50 */
        --success-green: #22c55e; /* green-500 */
        --error-red: #ef4444;    /* red-500 */
        --bg-white: #ffffff;
        --bg-zinc-50: #f9fafb;
    }
    
    body, .stApp {
        font-family: 'Be Vietnam Pro', sans-serif;
        background: var(--bg-zinc-50);
    }
    
    .stButton>button {
        background: var(--primary-blue);
        color: white;
        border: 1px solid var(--primary-blue);
    }
    .stButton>button:hover {
        background: var(--hover-blue);
    }
    
    .stTextInput>div>div>input {
        border: 1px solid var(--primary-blue);
        background: var(--bg-white);
    }
    
    .stAlert {
        background: var(--light-blue);
        color: var(--primary-blue);
    }
    
    /* Gradient cho header */
    header {
        background: linear-gradient(to bottom right, #3b82f6, #6366f1); /* blue-500 to indigo-600 */
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

st.title("RAG Chatbot - Ollama + Streamlit")
st.markdown("Hỗ trợ PDF/TXT/DOCX từ nhiều folder. Chat dựa trên tài liệu.")

# Session state cho chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_loaded" not in st.session_state:
    st.session_state.vector_loaded = False

# Input folders
with st.sidebar:
    st.header("Tải tài liệu")
    folders_input = st.text_area("Nhập paths folders (mỗi dòng 1 path, hỗ trợ subfolders):", height=100)
    if st.button("Tải và index tài liệu"):
        if folders_input:
            folders = [f.strip() for f in folders_input.split("\n") if f.strip()]
            try:
                response = requests.post(f"{API_URL}/index_documents", json={"folders": folders})
                if response.status_code == 200:
                    st.success("Tài liệu đã index thành công!")
                    st.session_state.vector_loaded = True
                else:
                    st.error(f"Lỗi: {response.json().get('detail')}")
            except Exception as e:
                st.error(f"Kết nối server lỗi: {str(e)}")
        else:
            st.warning("Vui lòng nhập ít nhất 1 folder.")

# Chat interface (chỉ hiển thị nếu vector loaded)
if st.session_state.vector_loaded:
    # Hiển thị history
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.chat_message("user").markdown(msg["content"])
        else:
            st.chat_message("assistant").markdown(msg["content"])

    # Input question
    question = st.chat_input("Hỏi gì đó về tài liệu...")
    if question:
        # Lưu history trước khi thêm message mới
        previous_history = st.session_state.chat_history.copy()
        st.session_state.chat_history.append({"role": "user", "content": question})
        try:
            response = requests.post(f"{API_URL}/query", json={
                "question": question,
                "chat_history": previous_history  # Chỉ gửi history trước đó
            })
            if response.status_code == 200:
                ai_response = response.json()["response"]
                st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
                st.rerun()  # Refresh để hiển thị
            else:
                st.error(f"Lỗi: {response.json().get('detail')}")
        except Exception as e:
            st.error(f"Kết nối server lỗi: {str(e)}")
else:
    st.info("Tải tài liệu trước để chat.")
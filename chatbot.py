import streamlit as st
import numpy as np
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings

# -------- LLM & EMBEDDINGS --------
llm = Ollama(model="llama3.2:1b", temperature=0.3)
embeddings_model = OllamaEmbeddings(model="nomic-embed-text")


# -------- UTILS --------
def extract_text_from_files(uploaded_files):
    combined_text = ""
    for f in uploaded_files:
        if f.name.endswith(".pdf"):
            reader = PdfReader(f)
            for page in reader.pages:
                if page.extract_text():
                    combined_text += page.extract_text() + "\n"
        elif f.name.endswith(".txt"):
            combined_text += f.read().decode("utf-8") + "\n"
    return combined_text


def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
    return splitter.split_text(text)


def embed_text(texts):
    return embeddings_model.embed_documents(texts)


def embed_query(query):
    return embeddings_model.embed_query(query)


def cosine(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def retrieve(query, chunks, chunk_embeddings, top_k=3):
    q_emb = embed_query(query)
    scores = [(cosine(q_emb, emb), i) for i, emb in enumerate(chunk_embeddings)]
    scores.sort(reverse=True)
    return "\n\n".join([chunks[i] for _, i in scores[:top_k]])


def rag_answer(context, question):
    prompt = f"""
Use ONLY the context below to answer.

Context:
{context}

Question:
{question}

Give a short and accurate answer.
"""
    return llm.invoke(prompt)


# -------- STREAMLIT APP --------
def main():
    st.title("📄 RAG Chatbot (Ollama + LangChain)")

    uploaded_files = st.sidebar.file_uploader(
        "Upload Files", type=["pdf", "txt"], accept_multiple_files=True
    )

    if "chunks" not in st.session_state:
        st.session_state.chunks = []
    if "chunk_embeddings" not in st.session_state:
        st.session_state.chunk_embeddings = []

    if uploaded_files:
        with st.spinner("Processing documents..."):
            text = extract_text_from_files(uploaded_files)
            if text.strip():
                st.session_state.chunks = chunk_text(text)
                st.session_state.chunk_embeddings = embed_text(st.session_state.chunks)
                st.success(f"✅ Indexed {len(st.session_state.chunks)} chunks")

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi! Upload a file and ask anything 😊"}
        ]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_q = st.chat_input("Ask your question...")
    if user_q:
        if not st.session_state.chunks:
            st.warning("⚠️ Upload a file first")
            return

        st.session_state.messages.append({"role": "user", "content": user_q})
        with st.chat_message("user"):
            st.markdown(user_q)

        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                context = retrieve(
                    user_q, st.session_state.chunks, st.session_state.chunk_embeddings
                )
                answer = rag_answer(context, user_q)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )
                st.markdown(answer)


if __name__ == "__main__":
    main()

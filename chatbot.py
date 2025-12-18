import streamlit as st
import numpy as np
import os
import time
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from google import genai
from google.genai import types
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def rate_limit_handler(func, *args, **kwargs):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str or "quota" in error_str.lower():
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 10
                    st.warning(
                        f"⏳ Rate limit hit. Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                else:
                    raise Exception(
                        "⚠️ Rate limit exceeded. Please wait 1-2 minutes and try again.")
            else:
                raise e


def embed_text(texts):
    embeddings = []
    for i, t in enumerate(texts):
        try:
            response = rate_limit_handler(
                client.models.embed_content,
                model="models/text-embedding-004",
                contents=t
            )
            embeddings.append(response.embeddings[0].values)
            if i > 0 and i % 10 == 0:
                time.sleep(2)
                st.info(f"📊 Processed {i}/{len(texts)} chunks...")
        except Exception as e:
            st.error(f"❌ Error embedding chunk {i}: {str(e)}")
            raise e
    return embeddings


def embed_query(query):
    response = rate_limit_handler(
        client.models.embed_content,
        model="models/text-embedding-004",
        contents=query
    )
    return response.embeddings[0].values


def extract_text_from_files(uploaded_files):
    combined_text = ""
    for f in uploaded_files:
        if f.name.endswith('.pdf'):
            reader = PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    combined_text += page_text + "\n"
        elif f.name.endswith('.txt'):
            combined_text += f.read().decode("utf-8") + "\n"
    return combined_text


def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=300
    )
    return splitter.split_text(text)


def cosine(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def retrieve(query, chunks, chunk_embeddings, top_k=3):
    q_emb = embed_query(query)
    scores = []
    for i, embd in enumerate(chunk_embeddings):
        score = cosine(q_emb, embd)
        scores.append((score, i))
    scores.sort(reverse=True)
    return "\n\n".join([chunks[idx] for _, idx in scores[:top_k]])


def rag_answer(context, question):
    prompt = f"""Use ONLY the context below to answer the question.
Context:
{context}
Question:
{question}
Give a short and accurate answer."""
    response = rate_limit_handler(
        client.models.generate_content,
        model="gemini-2.0-flash",
        contents=prompt
    )
    return response.text


def main():
    st.title("ChatBot")
    uploaded_files = st.sidebar.file_uploader(
        "Upload Files",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )
    if "chunks" not in st.session_state:
        st.session_state.chunks = []
    if "chunk_embeddings" not in st.session_state:
        st.session_state.chunk_embeddings = []
    if uploaded_files:
        current_file_ids = [f.name + str(f.size) for f in uploaded_files]
        if "processed_ids" not in st.session_state or st.session_state.processed_ids != current_file_ids:
            with st.spinner("Processing documents..."):
                st.session_state.processed_ids = current_file_ids
                text = extract_text_from_files(uploaded_files)
                if text.strip():
                    st.session_state.chunks = chunk_text(text)
                    try:
                        st.session_state.chunk_embeddings = embed_text(
                            st.session_state.chunks)
                        st.success(
                            f"✅ Indexed {len(st.session_state.chunks)} chunks!")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                else:
                    st.error("❌ No text found.")
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi! Upload a file and ask anything 😊"}]
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    user_q = st.chat_input("Ask your question...")
    if user_q:
        if not uploaded_files:
            st.warning("⚠️ Please upload a file first!")
        elif not st.session_state.chunks:
            st.warning("⚠️ No text available.")
        else:
            st.session_state.messages.append(
                {"role": "user", "content": user_q})
            with st.chat_message("user"):
                st.markdown(user_q)
            with st.chat_message("assistant"):
                with st.spinner("🤔 Thinking..."):
                    try:
                        context = retrieve(
                            user_q, st.session_state.chunks, st.session_state.chunk_embeddings)
                        answer = rag_answer(context, user_q)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": answer})
                        st.markdown(answer)
                    except Exception as e:
                        st.error(f"Error: {str(e)}")


if __name__ == "__main__":
    main()

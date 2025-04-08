import os
import re
import faiss
import numpy as np
import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# Initialize the Google Gemini API
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite-preview-02-05", api_key="YOUR_KEY")

# FAISS Index Path
vector_db_path = "faiss_index"

# HuggingFace Embeddings Model
embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Function to extract video ID from a YouTube URL
def extract_video_id(url):
    match = re.search(r"(?<=v=)[\w-]+|(?<=youtu\.be/)[\w-]+", url)
    return match.group(0) if match else None

# Function to fetch YouTube subtitles
def get_youtube_subtitles(video_url):
    video_id = extract_video_id(video_url)
    if not video_id:
        return None

    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        subtitles = "\n".join([entry["text"] for entry in transcript])
        return subtitles
    except Exception as e:
        st.error(f"Error fetching subtitles: {e}")
        return None

# Load subtitles for multiple YouTube videos
def load_youtube_subtitles(video_urls):
    documents = []
    for url in video_urls:
        subtitles = get_youtube_subtitles(url)
        if subtitles:
            documents.append(subtitles)
    return documents

# Function to compute and store embeddings
def compute_embeddings(video_urls):
    documents = load_youtube_subtitles(video_urls)
    if not documents:
        st.error("No subtitles found.")
        return

    embeddings = embedder.embed_documents(documents)
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings, dtype=np.float32))
    faiss.write_index(index, vector_db_path)
    st.success("Embeddings stored successfully!")

# Function to process a query using FAISS and Gemini
def process_query(query, video_urls, chat_history):
    if not os.path.exists(vector_db_path):
        st.error("FAISS index not found. Please generate embeddings first.")
        return

    index = faiss.read_index(vector_db_path)
    query_embedding = np.array([embedder.embed_query(query)], dtype=np.float32)
    _, indices = index.search(query_embedding, k=3)

    video_subtitles = load_youtube_subtitles(video_urls)
    retrieved_docs = [video_subtitles[i] for i in indices[0] if i < len(video_subtitles)]
    context = "\n\n".join(retrieved_docs)

    full_context = f"{chat_history}\n\nContext:\n{context}\n\nQuery: {query}" if chat_history else f"Context:\n{context}\n\nQuery: {query}"

    message = HumanMessage(content=full_context)
    response = llm.invoke([message])

    return response.content

# Streamlit UI
st.title("🎬 YouTube Subtitle Chatbot")
st.sidebar.header("📌 Enter YouTube URLs")
video_urls = st.sidebar.text_area("Enter one URL per line").split("\n")
video_urls = [url.strip() for url in video_urls if url.strip()]

if st.sidebar.button("Generate Embeddings"):
    if video_urls:
        compute_embeddings(video_urls)
    else:
        st.sidebar.warning("Please enter at least one YouTube URL.")

if video_urls:
    st.subheader("🎥 Watch Videos")
    for url in video_urls:
        video_id = extract_video_id(url)
        if video_id:
            st.video(f"https://www.youtube.com/embed/{video_id}")

# Chatbot UI
st.header("💬 Chat with YouTube Videos")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("Ask a question about the video subtitles:")

if st.button("Send"):
    if user_input and video_urls:
        response = process_query(user_input, video_urls, "\n".join(st.session_state.chat_history))

        if response:
            st.session_state.chat_history.append(f"**👤 You:** {user_input}")
            st.session_state.chat_history.append(f"**🤖 AI:** {response}")

st.markdown("### 📝 Chat History")
for message in st.session_state.chat_history:
    if message.startswith("**👤 You:**"):
        st.markdown(f'<p style="color:blue; font-weight:bold;">{message}</p>', unsafe_allow_html=True)
    else:
        st.markdown(f'<p style="color:green; font-weight:bold;">{message}</p>', unsafe_allow_html=True)

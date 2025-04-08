import os
import json
import faiss
import numpy as np
import streamlit as st
from PyPDF2 import PdfReader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite-preview-02-05", api_key="YOUR_KEY")
embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db_path = "faiss_index"

def load_pdfs(uploaded_files):
    documents = []
    sources = []
    for uploaded_file in uploaded_files:
        reader = PdfReader(uploaded_file)
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        documents.append(text)
        sources.append(uploaded_file.name) 
    return documents, sources

# Compute and store embeddings
def compute_embeddings(uploaded_files):
    documents, sources = load_pdfs(uploaded_files)
    if not documents:
        st.warning("No PDFs found. Please upload documents.")
        return
    
    embeddings = embedder.embed_documents(documents)
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings, dtype=np.float32))
    faiss.write_index(index, vector_db_path)

    # Save document sources for citation tracking
    with open("sources.json", "w") as f:
        json.dump(sources, f)

    st.success("Embeddings stored in FAISS!")

# Process query and retrieve answers with citations
def process_query(query, chat_history):
    if not os.path.exists(vector_db_path):
        st.warning("FAISS index not found. Please upload and process PDFs first.")
        return "", []

    index = faiss.read_index(vector_db_path)
    query_embedding = np.array([embedder.embed_query(query)], dtype=np.float32)
    _, indices = index.search(query_embedding, k=3)

    # Load document texts and sources
    documents, sources = load_pdfs(uploaded_files)
    with open("sources.json", "r") as f:
        saved_sources = json.load(f)

    # Retrieve relevant documents and their sources
    retrieved_docs = [(documents[i], saved_sources[i]) for i in indices[0] if i < len(documents)]
    context = "\n\n".join([doc for doc, _ in retrieved_docs])
    cited_sources = list(set([src for _, src in retrieved_docs])) 

    full_context = f"{chat_history}\n\nContext:\n{context}\n\nQuery: {query}" if chat_history else f"Context:\n{context}\n\nQuery: {query}"
    message = HumanMessage(content=full_context)
    response = llm.invoke([message])

    return response.content, cited_sources

# Streamlit UI
st.title("📄 PDF Chatbot with FAISS and Gemini")

# Sidebar for uploading PDFs
st.sidebar.header("📤 Upload PDFs")
uploaded_files = st.sidebar.file_uploader("Upload PDF documents", accept_multiple_files=True, type=["pdf"])

if st.sidebar.button("🔄 Process Documents"):
    if uploaded_files:
        compute_embeddings(uploaded_files)
    else:
        st.sidebar.warning("Please upload at least one PDF.")

# Display uploaded PDFs in the UI
if uploaded_files:
    st.subheader("📑 Uploaded PDFs")
    for uploaded_file in uploaded_files:
        st.markdown(f"📄 **{uploaded_file.name}**")
        with st.expander(f"Preview: {uploaded_file.name}"):
            reader = PdfReader(uploaded_file)
            preview_text = "\n".join([page.extract_text() for page in reader.pages[:3] if page.extract_text()])
            st.text_area("Preview (First 3 Pages)", preview_text, height=150)

# Chatbot interface
st.header("💬 Chat with Your PDFs")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = ""

query = st.text_input("Ask a question about your documents:")

if st.button("Send") and query:
    response, citations = process_query(query, st.session_state.chat_history)

    if response:
        st.session_state.chat_history += f"\n**👤 You:** {query}\n**🤖 AI:** {response}"
        
        # Display chat history
        st.markdown("### 📝 Chat History")
        for message in st.session_state.chat_history.split("\n"):
            if message.startswith("**👤 You:**"):
                st.markdown(f'<p style="color:blue; font-weight:bold;">{message}</p>', unsafe_allow_html=True)
            elif message.startswith("**🤖 AI:**"):
                st.markdown(f'<p style="color:green; font-weight:bold;">{message}</p>', unsafe_allow_html=True)
        
        # Display citations
        if citations:
            st.markdown("### 🔗 Citations")
            for cite in citations:
                st.markdown(f"- 📌 **{cite}**")
        
        # Display AI response
        st.success(response)

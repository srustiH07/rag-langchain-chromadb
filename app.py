import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

st.set_page_config(
    page_title="Cynaris RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 RAG Pipeline with LangChain & ChromaDB")
st.caption("Cynaris AI/ML Internship Project")

PDF_PATH = "data/rag_demo_document.pdf"
CHROMA_PATH = "chroma_db"

@st.cache_resource
def create_vectorstore():
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="rag_documents",
        persist_directory=CHROMA_PATH
    )

    return vectorstore


vectorstore = create_vectorstore()

st.success("Knowledge base ready!")

st.markdown("### Ask a question")

question = st.text_input(
    "Enter your question:",
    placeholder="Example: What is Retrieval-Augmented Generation?"
)

if question:
    results = vectorstore.similarity_search(question, k=4)

    if results:
        st.markdown("### Answer")

        # Retrieval-based answer used when Groq API key is unavailable
        answer = results[0].page_content

        st.write(answer)

        st.markdown("### 📚 Sources")

        sources = set()

        for doc in results:
            source = doc.metadata.get("source", "Unknown source")
            page = doc.metadata.get("page", 0) + 1
            sources.add(f"{source} — Page {page}")

        for source in sources:
            st.write(f"- {source}")

        with st.expander("🔎 Retrieved Context"):
            for i, doc in enumerate(results, 1):
                st.markdown(f"**Chunk {i}**")
                st.write(doc.page_content)
                st.divider()

st.sidebar.title("Project Components")

st.sidebar.markdown("""
✅ PDF Document Loader

✅ Text Chunking

✅ Sentence Transformer Embeddings

✅ ChromaDB Vector Store

✅ Semantic Retrieval

✅ Source Citations

⏳ Groq LLM Integration

---

**Cynaris AI/ML Internship**

RAG Pipeline Project
""")
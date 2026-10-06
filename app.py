import streamlit as st
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# --------------------------------------------------
# Project paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
PDF_PATH = BASE_DIR / "data" / "rag_demo_document.pdf"
CHROMA_DIR = BASE_DIR / "chroma_db"

# --------------------------------------------------
# Page
# --------------------------------------------------

st.set_page_config(
    page_title="Cynaris RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 RAG Pipeline with LangChain & ChromaDB")
st.caption("Cynaris AI/ML Internship Project")

# --------------------------------------------------
# Load and index documents
# --------------------------------------------------

@st.cache_resource
def create_vectorstore():

    loader = PyPDFLoader(str(PDF_PATH))
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
        persist_directory=str(CHROMA_DIR),
        collection_name="rag_documents"
    )

    return vectorstore


# --------------------------------------------------
# Start
# --------------------------------------------------

if not PDF_PATH.exists():
    st.error("PDF document not found in the data folder.")
    st.stop()

with st.spinner("Loading document and building knowledge base..."):
    vectorstore = create_vectorstore()

st.success("Knowledge base ready!")

# --------------------------------------------------
# Question
# --------------------------------------------------

question = st.text_input(
    "Ask a question about the document:",
    placeholder="Example: What is RAG?"
)

if question:

    with st.spinner("Searching the knowledge base..."):

        results = vectorstore.similarity_search(
            question,
            k=4
        )

    st.subheader("Answer")

    if results:

        # Use retrieved text as a grounded answer
        answer = results[0].page_content

        st.write(answer)

        st.subheader("📚 Sources")

        seen = set()

        for doc in results:

            source = Path(
                doc.metadata.get("source", "Unknown")
            ).name

            page = doc.metadata.get("page", 0) + 1

            source_name = f"{source} — Page {page}"

            if source_name not in seen:

                st.write(f"• {source_name}")

                seen.add(source_name)

        st.subheader("🔎 Retrieved Context")

        for i, doc in enumerate(results, 1):

            with st.expander(f"Retrieved Chunk {i}"):

                st.write(doc.page_content)

else:

    st.info(
        "Enter a question above to search the document."
    )

# --------------------------------------------------
# Project information
# --------------------------------------------------

with st.sidebar:

    st.header("Project Components")

    st.write("✅ PDF Document Loader")
    st.write("✅ Text Chunking")
    st.write("✅ Sentence Transformers")
    st.write("✅ ChromaDB")
    st.write("✅ Semantic Retrieval")
    st.write("✅ Source Citations")
    st.write("⏳ Groq LLM integration")

    st.divider()

    st.write("Cynaris Internship")
    st.write("RAG Pipeline Project")
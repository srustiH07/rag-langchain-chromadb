# RAG Pipeline with LangChain & ChromaDB

## Cynaris AI/ML Internship Project

A Retrieval-Augmented Generation (RAG) pipeline built using LangChain, ChromaDB, Sentence Transformers, and Streamlit.

## Project Overview

This project demonstrates how documents can be processed, converted into vector embeddings, stored in a vector database, and retrieved based on the semantic similarity of a user's question.

The pipeline is designed as the foundation for a question-answering chatbot with source citations.

## Architecture

PDF Document
    ↓
Document Loading
    ↓
Text Chunking
    ↓
Sentence Transformer Embeddings
    ↓
ChromaDB Vector Store
    ↓
Semantic Retrieval
    ↓
Retrieved Context
    ↓
Answer + Source Citation

## Technologies Used

- Python 3.11
- LangChain
- ChromaDB
- Sentence Transformers
- Hugging Face Embeddings
- Streamlit
- PyPDF
- ReportLab

## Key Features

- PDF document ingestion
- Automatic text extraction
- Recursive text chunking
- Semantic embeddings using `all-MiniLM-L6-v2`
- ChromaDB vector storage
- Semantic similarity retrieval
- Source/page citations
- Streamlit-based user interface
- Retrieved context display

## Project Structure

```text
rag-langchain-chromadb/
│
├── app.py
├── create_pdf.py
├── README.md
├── .gitignore
│
└── data/
    └── rag_demo_document.pdf
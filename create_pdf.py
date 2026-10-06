from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from pathlib import Path

output = Path("data/rag_demo_document.pdf")

doc = SimpleDocTemplate(str(output), pagesize=A4)
styles = getSampleStyleSheet()

content = [
    ("RAG Pipeline: A Practical Guide", "Title"),
    ("Introduction to Retrieval-Augmented Generation", "Heading1"),
    ("Retrieval-Augmented Generation (RAG) combines information retrieval with a large language model. "
     "A RAG application retrieves relevant passages from a document collection and provides those passages "
     "to the language model as context before generating an answer.", "BodyText"),

    ("Document Processing", "Heading1"),
    ("Documents such as PDF files can be divided into smaller text chunks. Chunking makes retrieval more "
     "precise because the vector database can return focused sections instead of an entire document. "
     "Chunk size and overlap are important because very small chunks may lose context while very large "
     "chunks may contain unrelated information.", "BodyText"),

    ("Embeddings and ChromaDB", "Heading1"),
    ("An embedding model converts text into numerical vectors. Texts with similar meanings tend to have "
     "vectors that are close together. ChromaDB is a vector database that can store these embeddings along "
     "with the original text and metadata such as the source filename, page number, and chunk identifier.", "BodyText"),

    ("LangChain and Groq", "Heading1"),
    ("LangChain connects document loaders, text splitters, embedding models, vector stores, retrievers, "
     "prompts, and language models. Groq provides access to language models through an API. In this project, "
     "the Groq model receives the user's question together with retrieved document context and generates "
     "the final answer.", "BodyText"),

    ("Source Citations", "Heading1"),
    ("Source citations make a RAG chatbot more trustworthy. Retrieved chunks should retain metadata showing "
     "where they came from. The final response can display the document name and page or chunk information "
     "so users can verify the answer.", "BodyText"),

    ("Testing Questions", "Heading1"),
    ("1. What is RAG?\n"
     "2. Why is document chunking used?\n"
     "3. What does ChromaDB store?\n"
     "4. What is the role of LangChain?\n"
     "5. What is Groq used for?\n"
     "6. Why are source citations important?", "BodyText"),
]

story = []

for text, style in content:
    story.append(Paragraph(text, styles[style]))
    story.append(Spacer(1, 10))

doc.build(story)

print(f"PDF created successfully: {output}")
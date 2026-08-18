from pathlib import Path
from typing import List
from dotenv import load_dotenv
import os
import certifi
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
import docx2txt


load_dotenv()
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

Path("uploads").mkdir(exist_ok=True)
Path("chroma_db").mkdir(exist_ok=True)


#initializing embedding model
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")

#creating vector database like chroma or pinecone
vectorstore = Chroma(
    collection_name="agentic_chatbot_docs",
    persist_directory="./chroma_db",
    embedding_function=embeddings
)


def read_file_text(file_path:str) -> str:
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        reader = PdfReader(file_path)
        text = ""

        for page in reader.pages:
            text += page.extract_text() or ""
            text += "\n"

        return text

    if suffix == ".docx":
        return docx2txt.process(file_path)

    if suffix in [".txt", ".md",".py",".csv"]:
        return path.read_text(encoding="utf-8", errors="ignore")

    raise ValueError(f"Unsupported file type: Upload PDF, DOCX, TXT,MD,PY, or CSV")



#creating the vector store
def add_document_to_rag(file_path:str, thread_id:str) -> str:
    text = read_file_text(file_path)

    if not text.strip():
        raise ValueError("No text could be extracted from this file")

    spliter = RecursiveCharacterTextSplitter(chunk_size=900,chunk_overlap=150)
    chunks = spliter.split_text(text)

    docs: List[Document] = [
        Document(
            page_content=chunk,
            metadata={
                "thread_id": thread_id,
                "source": Path(file_path).name
            }
        )
        for chunk in chunks
    ]

    vectorstore.add_documents(docs)
    return {
        "filename": Path(file_path).name,
        "chunks": len(docs),
    }


def retrieve_from_rag(query:str, thread_id:str, k:int=4) -> str:
    docs = vectorstore.similarity_search(
        query=query,
        k=k,
        filter={"thread_id": thread_id}
    )

    if not docs:
        return "No Relevant Uploaded document found"

    results = []

    for i,doc in enumerate(docs, start=1):
        source = doc.metadata.get("source","uploade document")
        results.append(
            f"[Source {i}: {source}]\n{doc.page_content}"
        )

    return "\n".join(results)
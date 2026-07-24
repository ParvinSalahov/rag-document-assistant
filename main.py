import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# Load environment variables
load_dotenv()

# Verify OpenRouter / OpenAI API Key setup
api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")

def run_rag_pipeline():
    print("--- RAG Pipeline Execution ---")
    
    file_path = "sample_doc.txt"
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return

    # Checkpoint 1: Document Ingestion & Chunking
    print("\n[1] Ingesting Document & Creating Chunks...")
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=40,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"-> Successfully created {len(chunks)} chunks.")

    # Checkpoint 2: Embedding Generation & Vector Store Indexing
    print("\n[2] Generating Embeddings & Storing in ChromaDB...")
    
    # Initialize Embedding Model using OpenRouter API
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=api_key,
        openai_api_base="https://openrouter.ai/api/v1"
    )

    persist_directory = "./chroma_db"

    # Index chunks into Chroma VectorDB
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    print("-> Embeddings successfully generated and indexed in ChromaDB.")
    
    # Perform a similarity search test
    test_query = "What is the chunk size guideline?"
    print(f"\n[3] Testing Similarity Search for: '{test_query}'")
    results = vectorstore.similarity_search(test_query, k=2)
    
    for i, doc in enumerate(results):
        print(f"\nResult {i+1}:")
        print(doc.page_content)

if __name__ == "__main__":
    run_rag_pipeline()
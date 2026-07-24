import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

def run_checkpoint_1_ingestion():
    """
    Checkpoint 1: Document Ingestion and Chunking Strategy
    Reads a document and splits it into structured chunks with overlap.
    """
    file_path = "sample_doc.txt"
    
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return

    print("--- Checkpoint 1: Document Ingestion & Chunking ---")
    
    # 1. Load Document
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()
    print(f"Loaded {len(documents)} document(s).")

    # 2. Chunking Strategy (Recursive Character Splitter with Overlap)
    # Using chunk_size=200 and chunk_overlap=40 as per strategy guidelines
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=40,
        length_function=len,
        is_separator_regex=False,
    )
    
    chunks = text_splitter.split_documents(documents)
    
    print(f"Total Chunks Created: {len(chunks)}\n")
    
    # Display individual chunks to verify chunk boundary and overlap
    for idx, chunk in enumerate(chunks):
        print(f"--- Chunk {idx + 1} ---")
        print(f"Content: {chunk.page_content}")
        print(f"Length: {len(chunk.page_content)} chars\n")

if __name__ == "__main__":
    run_checkpoint_1_ingestion()
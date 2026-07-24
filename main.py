import os
from pathlib import Path

# LangChain imports
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# =====================================================================
# CHECKPOINT 1: READ OPENROUTER API KEY FROM .ENV FILE
# =====================================================================
BASE_DIR = Path(__file__).resolve().parent
env_path = BASE_DIR / ".env"

api_key = None
if env_path.exists():
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "OPENROUTER_API_KEY" in line:
                parts = line.split("=", 1)
                if len(parts) == 2:
                    api_key = parts[1].strip().strip('"\'')
                    break

if not api_key:
    print("❌ [CHECKPOINT 1 ERROR] OPENROUTER_API_KEY not found in .env file!")
    exit()

# Configure system environment variables for OpenRouter
os.environ["OPENAI_API_KEY"] = api_key
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

print("✅ [CHECKPOINT 1] OpenRouter API key loaded successfully.")


# =====================================================================
# CHECKPOINT 2: DOCUMENT LOADING
# =====================================================================
doc_path = BASE_DIR / "data.txt"

# If data.txt does not exist, create a sample file for testing
if not doc_path.exists():
    with open(doc_path, "w", encoding="utf-8") as f:
        f.write("DevJoint Intern RAG System: This system processes student tasks and generates responses using RAG architecture.")

loader = TextLoader(str(doc_path), encoding="utf-8")
documents = loader.load()
print(f"✅ [CHECKPOINT 2] Document loaded successfully. Total characters: {len(documents[0].page_content)}")


# =====================================================================
# CHECKPOINT 3: TEXT SPLITTING
# =====================================================================
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
splits = text_splitter.split_documents(documents)
print(f"✅ [CHECKPOINT 3] Document split into {len(splits)} chunks.")


# =====================================================================
# CHECKPOINT 4: EMBEDDINGS AND VECTOR STORE (ChromaDB)
# =====================================================================
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=api_key,
    openai_api_base="https://openrouter.ai/api/v1"
)

# Initialize in-memory vector database
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
print("✅ [CHECKPOINT 4] Vector store (ChromaDB) created and indexed successfully.")


# =====================================================================
# CHECKPOINT 5: LLM AND PROMPT TEMPLATE SETUP (RAG Chain)
# =====================================================================
llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    openai_api_key=api_key,
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0
)

# Prompt template enforcing context-based answers
template = """Answer the question based only on the following context.
If you do not know the answer, say that you don't know, do not try to make up an answer.

Context:
{context}

Question: {question}

Answer:"""

prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# LCEL (LangChain Expression Language) pipeline construction
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print("✅ [CHECKPOINT 5] RAG chain successfully constructed.")


# =====================================================================
# CHECKPOINT 6: RAG PIPELINE EXECUTION & TEST
# =====================================================================
if __name__ == "__main__":
    print("\n--- RAG PIPELINE EXECUTION TEST ---")
    query = "What does the DevJoint Intern RAG System do?"
    print(f"Question: {query}\n")
    
    response = rag_chain.invoke(query)
    print("🤖 Model Response:")
    print(response)
import os
from pathlib import Path

# LangChain imports
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# =====================================================================
# CHECKPOINT 1: READ OPENROUTER API KEY
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
    print("❌ [CHECKPOINT 1 ERROR] OPENROUTER_API_KEY not found!")
    exit()

os.environ["OPENAI_API_KEY"] = api_key
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

# =====================================================================
# CHECKPOINT 2 & 3: LOADING & SPLITTING
# =====================================================================
doc_path = BASE_DIR / "data.txt"
if not doc_path.exists():
    with open(doc_path, "w", encoding="utf-8") as f:
        f.write("DevJoint Intern RAG System: This system processes student tasks and generates responses using RAG architecture.")

loader = TextLoader(str(doc_path), encoding="utf-8")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = text_splitter.split_documents(documents)

# =====================================================================
# CHECKPOINT 4: EMBEDDINGS & VECTORSTORE
# =====================================================================
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=api_key,
    openai_api_base="https://openrouter.ai/api/v1"
)

vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# =====================================================================
# CHECKPOINT 5: GENERATION WITH SOURCE CITATION (MƏNBƏ İSTİNADI)
# =====================================================================
llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    openai_api_key=api_key,
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0
)

template = """Answer the question based only on the provided context.

Context:
{context}

Question: {question}

Answer:"""

prompt = ChatPromptTemplate.from_template(template)

def ask_rag_with_sources(query: str):
    # 1. Relevant chunk-ları axtarıb tapırıq
    retrieved_docs = retriever.invoke(query)
    
    # 2. Chunk-ların mətnini birşləşdiririk
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    
    # 3. LLM-ə müraciət edirik
    formatted_prompt = prompt.format(context=context_text, question=query)
    response = llm.invoke(formatted_prompt)
    
    print("\n--- RAG RESPONSE WITH SOURCE CITATIONS ---")
    print(f"Question: {query}")
    print(f"\nGenerated Answer:\n{response.content}")
    
    print("\n📌 Retrieved Sources / Chunks:")
    for idx, doc in enumerate(retrieved_docs, 1):
        source_file = doc.metadata.get("source", "Unknown")
        print(f"  [{idx}] Source: {source_file}")
        print(f"      Content snippet: {doc.page_content[:100]}...\n")

if __name__ == "__main__":
    ask_rag_with_sources("What does the DevJoint Intern RAG System do?")
import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Dokumentin yüklənməsi (Encoding xətalarının qarşısını almaq üçün try-except)
file_path = "sample_doc.txt" # və ya "requirements.txt"

print(f"[1] Fayl yüklənir: {file_path}")

try:
    # İlkin olaraq standart UTF-8 ilə oxumağa çalışırıq
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()
except UnicodeDecodeError:
    # Əgər UTF-16 formatındadırsa, avtomatik UTF-16 ilə oxuyur
    print("-> UTF-8 ilə oxumaq olmadı, UTF-16 ilə cəhd edilir...")
    loader = TextLoader(file_path, encoding="utf-16")
    documents = loader.load()

print(f"-> Sənəd uğurla yükləndi! Ümumi simvol sayı: {len(documents[0].page_content)}")

# 2. Text Splitting (Chunk-lara bölmə)
print("\n[2] Mətn chunk-lara bölünür...")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,       # Hər chunk-ın maksimum ölçüsü
    chunk_overlap=50,     # Chunk-lar arasında kəsişmə hissəsi
    length_function=len
)

chunks = text_splitter.split_documents(documents)

print(f"-> Cəmi {len(chunks)} ədəd chunk yaradıldı.")

# 3. Nümunə kimi ilk chunk-ı ekrana çıxarırıq
if chunks:
    print("\n--- İlk Chunk Nümunəsi ---")
    print(chunks[0].page_content)
    print("----------------------------")
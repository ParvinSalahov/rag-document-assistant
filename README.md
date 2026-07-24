Markdown
# Sənəd Əsaslı Suallara Cavab Verən RAG Sistemi (Həftə 2)

Bu layihə təcrübə proqramının (internship) 2-ci həftə tapşırığı çərçivəsində yaradılmışdır. Layihədə LangChain, ChromaDB vektor bazası və OpenRouter API (GPT-4o-mini) vasitəsilə sıfırdan kamil RAG (Retrieval-Augmented Generation) sistemi reallaşdırılmışdır.

---

## Layihənin Quraşdırılması və İşə Salınması

Proqramı öz lokal kompüterinizdə işə salmaq üçün aşağıdakı addımları növbə ilə yerinə yetirin:

### 1. Virtual Mühitin (venv) Yaradılması və Aktivləşdirilməsi
Layihə qovluğunda terminalı açın və sisteminizə uyğun olaraq aşağıdakı əmrləri icra edin:

* **Virtual mühitin yaradılması:**
  ```bash
  python -m venv venv
Aktivləşdirilməsi (Windows - PowerShell üçün):

PowerShell
.\venv\Scripts\Activate.ps1
Aktivləşdirilməsi (Windows - CMD üçün):

DOS
venv\Scripts\activate
Aktivləşdirilməsi (macOS / Linux üçün):

Bash
source venv/bin/activate
(Aktivləşdikdən sonra terminal sətirinin əvvəlində (venv) yazısı görünməlidir).

2. Lazımi Kitabxanaların Quraşdırılması
Virtual mühit daxilində RAG sistemi üçün tələb olunan paketləri yükləmək üçün bu əmri yazın:

Bash
pip install langchain langchain-community langchain-openai langchain-text-splitters chromadb python-dotenv
3. API Açarlarının Konfiqurasiyası (.env)
Layihə qovluğunda .env adlı fayl yaradın və OpenRouter platformasından aldığınız API açarını təhlükəsiz şəkildə daxil edin:

Code snippet
OPENROUTER_API_KEY=sizin_real_openrouter_api_acariniz
(Qeyd: .gitignore faylı vasitəsilə real .env faylının GitHub-a sızmasının qarşısı təhlükəsiz şəkildə alınmışdır).

4. Kodu İşə Salın
Hər şey hazır olduqdan sonra RAG boru kəmərini (pipeline) sınaqdan keçirmək üçün proqramı başladın:

Bash
python main.py
📄 Checkpoint-lər və Nümunə Sorğu/Cavab Logları (Logs)
Checkpoint 1 & 2: Mühit və Sənəd Yükləmə (Document Loading)
API açarı .env faylından avtomatik oxunur.

TextLoader vasitəsilə data.txt sənədi mühitə yüklənir.

Checkpoint 3 & 4: Chunking və Vektor Bazası (ChromaDB)
Sənəd RecursiveCharacterTextSplitter ilə parçalanır (chunk_size=500, chunk_overlap=50).

text-embedding-3-small modeli vasitəsilə vektorlaşdırılaraq lokal ChromaDB bazasına indekslənir.

Checkpoint 5: Mənbə İstinadı ilə Cavab Generasiyası (Source Citation)
Sistem cavab generasiya edərkən istifadə etdiyi chunk-ların mənbəsini və daxilindəki mətni loglayır:

Real Terminal Çıxışı (Console Log):

Plaintext
(venv) PS C:\Users\HUAWEI\Desktop\devjoint_intern\week2_rag_system> python main.py

--- RAG RESPONSE WITH SOURCE CITATIONS ---
Question: What does the DevJoint Intern RAG System do?

Generated Answer:
The DevJoint Intern RAG System processes student tasks and generates responses using RAG architecture.

📌 Retrieved Sources / Chunks:
  [1] Source: C:\Users\HUAWEI\Desktop\devjoint_intern\week2_rag_system\data.txt
      Content snippet: DevJoint Intern RAG System: This system processes student tasks and generates responses using RAG ar...
🛡️ "Sənədlərdə Yoxdur" Halının İdarə Olunması (Checkpoint 6 - Fallback)
RAG sistemində LLM-in hallüsinasiya etməsinin (özündən uydurma məlumatlar verməsinin) qarşısını almaq üçün aşağıdakı tədbirlər görülmüşdür:

Prompt Engineering: System prompt vasitəsilə modelə daxil olan sorğunun cavabı verilən kontekstdə yoxdursa, uydurma cavab vermək əvəzinə dəqiq şəkildə "Bu məlumat sənədlərdə tapılmadı" demək şərti qoyulmuşdur.

Low Temperature: Model konfiqurasiyasında temperature=0 seçilmişdir ki, model determinik davransın.

📂 Layihə Strukturunun Görünüşü
Plaintext
week2_rag_system/
│
├── venv/                 # Virtual mühit qovluğu (GitHub-a yüklənilmir)
├── .env                  # Real API açarının saxlandığı gizli fayl (GitHub-a yüklənilmir)
├── .gitignore            # Hansı faylların Git-ə gitməyəcəyini təyin edən tənzimləmə
├── data.txt              # RAG sisteminin oxuduğu baza mətn sənədi
├── README.md             # Layihə haqqında ümumi məlumat və təlimat faylı
└── main.py               # Bütün RAG checkpoint-lərini icra edən əsas Python kodu
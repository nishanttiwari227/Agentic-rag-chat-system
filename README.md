#  🤖 Agentic RAG-based AI Chat System

A local Retrieval-Augmented Generation (RAG) based AI chat application that
allows users to ask questions from PDFs, documents, and web pages using
semantic search and large language models.

---

## 🚀 Features
- Document-based conversational AI (PDF, DOCX, TXT, Web URLs)
- FAISS-powered vector similarity search with HuggingFace embeddings
- Chunking and context injection to improve answer accuracy
- Multimodal input support (text, audio, images)
- Export chat responses to PDF, PPT, and DOC formats
- Project-based local persistence using SQLite and FAISS

---

## 🛠 Tech Stack
- **Language:** Python  
- **Framework:** Streamlit  
- **Vector Store:** FAISS  
- **Embeddings:** HuggingFace (sentence-transformers)  
- **LLM:** Google Gemini API  
- **Orchestration:** LangChain  
- **Database:** SQLite  

---

## 📂 Project Structure
```text
agentic-rag-chat-system/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── assets/ (optional screenshots)

```

## ▶️ How to Run Locally

- Clone the repository

```text
git clone https://github.com/your-username/agentic-rag-chat-system.git
cd agentic-rag-chat-system
```

- Create a virtual environment (optional but recommended)

```text
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

- Install dependencies

```text
pip install -r requirements.txt
```

- Add your Gemini API key
```text
Create a file .streamlit/secrets.toml

GEMINI_API_KEY = "your_api_key_here"
```

- Run the application

```text
streamlit run app.py
```

## 📌 Notes

```text
This project runs locally and does not upload documents to any external server.

Only relevant document chunks are sent to the LLM for response generation.

Designed as a prototype suitable for internships, demos, and learning RAG systems.
```

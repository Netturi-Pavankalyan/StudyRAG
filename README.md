# 📚 StudyRAG — AI-Powered Smart Study Companion

> Transform the way you learn, revise, and prepare for exams using the power of Generative AI and RAG (Retrieval-Augmented Generation).

## 🌟 What is StudyRAG?

**StudyRAG** is an AI-powered study companion that lets students chat with their own documents, auto-generate exam questions, and get accurate, document-specific answers — not generic internet responses. Built on a RAG (Retrieval-Augmented Generation) pipeline, every answer is grounded in *your* uploaded content.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 📄 **Multi-Format Upload** | Supports PDF, DOCX, TXT, and handwritten/image-based notes |
| 💬 **Chat with Documents** | Ask anything and get instant, context-aware answers from your files |
| 🧠 **Q&A Generation** | Auto-generate exam-style questions with detailed answers |
| 📝 **MCQ & Fill-in-the-Blanks** | Generate multiple question types for comprehensive exam prep |
| 🌐 **Web Search Fallback** | Falls back to web search when your documents don't have the answer |
| 🤖 **OCR / Vision AI** | Reads handwritten and scanned image-based PDFs |
| 🔐 **JWT Authentication** | Secure personal study space with JWT + bcrypt |
| ⚡ **Async Processing** | Real-time progress tracking for question generation |
| 📱 **Fully Responsive** | Mobile-first design that works on all screen sizes |
| 📥 **PDF Export** | Download generated Q&A as a PDF for offline study |

---

## 🔍 How RAG Works in StudyRAG

```
┌─────────────────┐     ┌──────────────────┐     ┌──────────────────────┐
│  Upload Document│────▶│  Chunk & Embed   │────▶│  Vector Store (Index)│
└─────────────────┘     └──────────────────┘     └──────────────────────┘
                                                            │
                                                            ▼
┌─────────────────┐     ┌──────────────────┐     ┌──────────────────────┐
│  Final Answer   │◀────│  Groq LLM        │◀────│  Retrieve Top-K Chunks│
│  (Grounded)     │     │  (Llama 3)       │     │  (Semantic Search)   │
└─────────────────┘     └──────────────────┘     └──────────────────────┘
```

**RAG (Retrieval-Augmented Generation)** retrieves the most relevant chunks from your uploaded documents and feeds them as context to the LLM, producing answers that are accurate, specific, and grounded in *your* material — not the open internet.

---

## 🛠️ Tech Stack

### Backend
- **FastAPI** — Async REST API framework
- **Python 3.10+** — Core language
- **Groq LLM (Llama 3)** — Large Language Model for generation
- **RAG Pipeline** — Custom retrieval-augmented generation
- **JWT + bcrypt** — Authentication & security
- **PostgreSQL** — Cloud database (async)

### Frontend
- **Vanilla HTML, CSS, JavaScript** — Lightweight, no framework overhead
- Mobile-first responsive design

### AI & ML
- **Groq API** — Blazing-fast LLM inference
- **Vision AI / OCR** — Handwritten & scanned document support
- **Sentence Embeddings** — Semantic chunking & retrieval

### Infrastructure
- **Render Cloud** — Deployment with 24/7 uptime monitoring
- **PostgreSQL (Cloud)** — Managed database

---

## 📁 Project Structure

```
StudyRAG/
├── main.py                  # FastAPI app entry point
├── requirements.txt
├── .env.example
│
├── routers/
│   ├── auth.py              # JWT authentication routes
│   ├── documents.py         # Upload & document management
│   ├── chat.py              # Chat with documents
│   └── questions.py         # Q&A, MCQ, fill-in-the-blanks generation
│
├── core/
│   ├── config.py            # App settings
│   ├── security.py          # JWT & bcrypt logic
│   └── database.py          # Async DB session
│
├── services/
│   ├── rag_pipeline.py      # RAG retrieval & generation
│   ├── ocr_service.py       # Vision AI / OCR
│   ├── web_search.py        # Web search fallback
│   └── pdf_export.py        # PDF export service
│
├── models/                  # SQLAlchemy models
├── schemas/                 # Pydantic schemas
│
└── frontend/
    ├── index.html
    ├── style.css
    └── app.js
```

---

## 🎓 Use Cases

- **Students** — Revise smarter by chatting with lecture notes and textbooks
- **Exam Prep** — Auto-generate practice questions from your own study material
- **Researchers** — Query large documents for specific insights instantly
- **Educators** — Create question banks from course material in seconds

---

## 🧠 What I Learned Building This

- Building production-ready async REST APIs with **FastAPI**
- Implementing **RAG pipelines** with LLMs for document QA
- **JWT authentication** & security best practices
- **OCR & Vision AI** integration for handwritten content
- Cloud deployment on **Render** with PostgreSQL
- Building **responsive, mobile-first** web applications

---

## 🤝 Contributing

Contributions are welcome! Here's how:

```bash
# 1. Fork the repo
# 2. Create a feature branch
git checkout -b feature/your-feature-name

# 3. Commit your changes
git commit -m "feat: add your feature"

# 4. Push and open a PR
git push origin feature/your-feature-name
```

Please follow [Conventional Commits](https://www.conventionalcommits.org/) for commit messages.

---

## 📄 License

This project is personal work by **Netturi Pavan Kalyan**. All rights reserved. You may not copy, distribute, or use this project without explicit permission from the author.

---

## 🙌 Acknowledgements

- [Groq](https://groq.com/) for ultra-fast LLM inference
- [FastAPI](https://fastapi.tiangolo.com/) for the amazing async framework
- [Llama 3](https://ai.meta.com/llama/) by Meta AI

---

## 📬 Contact

**Netturi Pavan Kalyan**  
[GitHub](https://github.com/Netturi-Pavankalyan) • [Live Demo](https://studyrag-l72m.onrender.com)

---

<p align="center">Made with ❤️ for students, by a student</p>

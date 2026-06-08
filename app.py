from fastapi import FastAPI, Request, UploadFile, File, HTTPException, Depends, status
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os, json, hashlib, uuid, re, io
from groq import Groq
from datetime import datetime
from pathlib import Path

# ── PDF READING (text + OCR fallback) ────────────────────────────────────────
try:
    from pypdf import PdfReader
    PDF_SUPPORT = True
except ImportError:
    try:
        from PyPDF2 import PdfReader
        PDF_SUPPORT = True
    except ImportError:
        PDF_SUPPORT = False

try:
    from pdf2image import convert_from_path
    PDF_TO_IMAGE_SUPPORT = True
except ImportError:
    PDF_TO_IMAGE_SUPPORT = False

try:
    import pytesseract
    OCR_SUPPORT = True
except ImportError:
    OCR_SUPPORT = False

# ── PDF GENERATION (styled output) ───────────────────────────────────────────
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, HRFlowable,
        Table, TableStyle, KeepTogether
    )
    from reportlab.lib.enums import TA_LEFT, TA_CENTER
    REPORTLAB_SUPPORT = True
except ImportError:
    REPORTLAB_SUPPORT = False

import base64
from PIL import Image as PILImage

# ── SESSION (simple in-memory, replace with Redis/JWT for production) ─────────
from starlette.middleware.sessions import SessionMiddleware

app = FastAPI(title="StudyRAG — Smart Study Assistant")

app.add_middleware(
    SessionMiddleware,
    secret_key=os.environ.get("SECRET_KEY", "smart-study-secret-key-2025"),
    max_age=86400,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ── In-memory stores (same as original Flask app) ────────────────────────────
users: dict = {}
documents: dict = {}
chat_history: dict = {}
cancelled_tasks: set = set()
_pdf_text_cache: dict = {}

ALLOWED_EXTENSIONS = {"pdf", "txt", "doc", "docx", "rtf", "odt"}
ALLOWED_MIMETYPES = {
    "application/pdf",
    "text/plain",
    "application/msword",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/rtf", "text/rtf",
    "application/vnd.oasis.opendocument.text",
}

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

GROQ_MODEL_PRIMARY  = "llama-3.3-70b-versatile"
GROQ_MODEL_FALLBACK = "llama-3.1-8b-instant"
GROQ_MODEL          = GROQ_MODEL_PRIMARY
GROQ_VISION_MODEL   = "meta-llama/llama-4-scout-17b-16e-instruct"

DOC_CONTEXT_CHARS  = 5000
CHAT_CONTEXT_CHARS = 5000

MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB


# ── Helper: session auth dependency ──────────────────────────────────────────
def get_current_user(request: Request) -> str:
    uid = request.session.get("user_id")
    if not uid:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return uid


# ── File validation ───────────────────────────────────────────────────────────
def allowed_file(filename: str, header: bytes | None = None) -> bool:
    if "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return False
    if header is not None:
        if header[:5] == b"%PDF-":
            return ext == "pdf"
        if header[:8] == b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1":
            return ext == "doc"
        if header[:4] == b"PK\x03\x04":
            return ext in ("docx", "odt")
        if ext in ("txt", "rtf"):
            return True
        return False
    return True


# ── PDF extraction helpers (unchanged logic) ─────────────────────────────────
def _pdf_to_images(path, dpi=200, max_pages=6):
    if not PDF_TO_IMAGE_SUPPORT:
        return []
    try:
        return convert_from_path(path, dpi=dpi, first_page=1, last_page=max_pages)
    except Exception as e:
        print("❌ PDF to image failed:", e)
        return []


def _image_to_b64(pil_img, max_width=1024):
    w, h = pil_img.size
    if w > max_width:
        pil_img = pil_img.resize((max_width, int(h * max_width / w)), PILImage.LANCZOS)
    buf = io.BytesIO()
    pil_img.convert("RGB").save(buf, format="JPEG", quality=82)
    return base64.b64encode(buf.getvalue()).decode()


def _vision_extract(path, max_pages=5):
    if not PDF_TO_IMAGE_SUPPORT:
        return ""
    try:
        images = _pdf_to_images(path, dpi=300, max_pages=max_pages)
        if not images:
            return ""
        parts = []
        for i, img in enumerate(images):
            b64 = _image_to_b64(img)
            try:
                resp = client.chat.completions.create(
                    model=GROQ_VISION_MODEL,
                    max_tokens=1500,
                    messages=[{"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                        {"type": "text", "text": (
                            f"Page {i+1}: Extract ALL readable text from this image. "
                            "This may contain handwritten notes or printed text. "
                            "Try your best to interpret unclear handwriting. "
                            "Do NOT summarize. Output the full text exactly as written."
                        )}
                    ]}]
                )
                page_text = resp.choices[0].message.content.strip()
                if page_text:
                    parts.append(f"[Page {i+1}]\n{page_text}")
            except Exception as page_err:
                err_str = str(page_err)
                if "429" in err_str and ("tokens per day" in err_str or "TPD" in err_str):
                    break
                continue
        return "\n\n".join(parts)
    except Exception as e:
        print(f"[vision_extract FAILED]: {e}")
        return ""


def _ocr_extract(path, max_chars=6000):
    if not OCR_SUPPORT or not PDF_TO_IMAGE_SUPPORT:
        return ""
    try:
        images = _pdf_to_images(path, dpi=300, max_pages=8)
        parts, total = [], 0
        for i, img in enumerate(images):
            t = pytesseract.image_to_string(img, config="--oem 3 --psm 6")
            parts.append(f"[Page {i+1}]\n{t}")
            total += len(t)
            if total >= max_chars:
                break
        return "\n\n".join(parts)[:max_chars]
    except Exception as e:
        print("[OCR FAILED]:", e)
        return ""


def _extract_txt(path, max_chars):
    try:
        with open(path, "r", errors="replace") as f:
            return f.read(max_chars)
    except Exception as e:
        print(f"[TXT extract FAILED]: {e}")
        return ""


def _extract_docx(path, max_chars):
    ext = path.rsplit(".", 1)[-1].lower()
    if ext == "docx":
        try:
            import docx as _docx
            doc = _docx.Document(path)
            return "\n".join(p.text for p in doc.paragraphs)[:max_chars]
        except Exception as e:
            print(f"[DOCX extract FAILED]: {e}")
    try:
        with open(path, "rb") as f:
            raw = f.read()
        chunks = re.findall(rb"[ -~]{4,}", raw)
        return "\n".join(c.decode("ascii", errors="ignore") for c in chunks)[:max_chars]
    except Exception as e:
        print(f"[DOC extract FAILED]: {e}")
        return ""


def _extract_rtf(path, max_chars):
    try:
        with open(path, "r", errors="replace") as f:
            raw = f.read()
        text = re.sub(r"\\[a-z]+[-\d]*[ ]?", " ", raw)
        text = re.sub(r"[{}\\]", "", text)
        return " ".join(text.split())[:max_chars]
    except Exception as e:
        print(f"[RTF extract FAILED]: {e}")
        return ""


def _extract_odt(path, max_chars):
    try:
        import zipfile, xml.etree.ElementTree as ET
        with zipfile.ZipFile(path) as z:
            with z.open("content.xml") as f:
                tree = ET.parse(f)
        parts = [el.text or "" for el in tree.iter() if el.text]
        return "\n".join(parts)[:max_chars]
    except Exception as e:
        print(f"[ODT extract FAILED]: {e}")
        return ""


def extract_pdf_text(path, max_chars=6000):
    text = ""
    if PDF_SUPPORT:
        try:
            reader = PdfReader(path)
            parts, total = [], 0
            for page in reader.pages:
                t = page.extract_text() or ""
                parts.append(t)
                total += len(t)
                if total >= max_chars:
                    break
            text = "\n".join(parts)[:max_chars]
        except Exception as e:
            print(f"[Tier1 FAILED]: {e}")
            text = ""

    if len(text.strip()) >= 300:
        return text

    vision_text = _vision_extract(path, max_pages=5)
    if vision_text.strip():
        return vision_text[:max_chars]

    ocr_text = _ocr_extract(path, max_chars=max_chars)
    if ocr_text.strip():
        return ocr_text

    return ""


def extract_text_from_file(path, max_chars=6000):
    ext = path.rsplit(".", 1)[-1].lower()
    if ext == "pdf":
        return extract_pdf_text(path, max_chars=max_chars)
    if ext == "txt":
        return _extract_txt(path, max_chars)
    if ext in ("doc", "docx"):
        return _extract_docx(path, max_chars)
    if ext == "rtf":
        return _extract_rtf(path, max_chars)
    if ext == "odt":
        return _extract_odt(path, max_chars)
    return ""


def _get_doc_full_text(doc):
    doc_id = doc["id"]
    if doc_id not in _pdf_text_cache:
        _pdf_text_cache[doc_id] = extract_text_from_file(
            doc["path"], max_chars=DOC_CONTEXT_CHARS * 10
        )
    return _pdf_text_cache[doc_id]


def get_doc_context(doc_id, user_id, max_chars=None):
    doc = next((d for d in documents.get(user_id, []) if d["id"] == doc_id), None)
    if not doc:
        return "", ""
    full_text = _get_doc_full_text(doc)
    if max_chars:
        return doc["name"], full_text[:max_chars]
    return doc["name"], full_text


def sanitise_messages(messages):
    clean, total = [], 0
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = m.get("role", "")
        content = str(m.get("content", ""))
        if role not in ("user", "assistant"):
            continue
        if len(content) > 2000:
            content = content[:2000] + "…"
        total += len(content)
        if total > 16000:
            break
        clean.append({"role": role, "content": content})
    return clean


# ── Rate limit error ──────────────────────────────────────────────────────────
class RateLimitError(Exception):
    def __init__(self, message, retry_after=None):
        super().__init__(message)
        self.retry_after = retry_after


def _parse_retry_after(err_str):
    m = re.search(r"(\d+)m(\d+(?:\.\d+)?)s", err_str)
    if m:
        return int(m.group(1)) * 60 + float(m.group(2))
    m = re.search(r"try again in (\d+(?:\.\d+)?)s", err_str)
    if m:
        return float(m.group(1))
    return None


def groq_chat(system, messages, max_tokens=2048):
    full_messages = [{"role": "system", "content": system}] + messages
    for model in (GROQ_MODEL_PRIMARY, GROQ_MODEL_FALLBACK):
        try:
            resp = client.chat.completions.create(
                model=model, max_tokens=max_tokens,
                messages=full_messages, temperature=0.7
            )
            return resp.choices[0].message.content
        except Exception as e:
            err_str = str(e)
            is_rate_limit = (
                "429" in err_str or "413" in err_str or
                "tokens per minute" in err_str or
                "rate_limit_exceeded" in err_str or
                "tokens per day" in err_str
            )
            if is_rate_limit:
                if model == GROQ_MODEL_FALLBACK:
                    retry = _parse_retry_after(err_str)
                    raise RateLimitError(err_str, retry_after=retry or 60)
                continue
            raise


# ── Styled PDF builder (unchanged logic) ─────────────────────────────────────
def build_styled_pdf(questions, doc_name):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=20*mm, rightMargin=20*mm,
        topMargin=22*mm, bottomMargin=22*mm,
    )
    W, H = A4

    PURPLE       = colors.HexColor("#6C4FF6")
    PURPLE_DARK  = colors.HexColor("#4930D4")
    PURPLE_LIGHT = colors.HexColor("#EDE9FE")
    TEAL         = colors.HexColor("#0DB884")
    TEAL_LIGHT   = colors.HexColor("#D1FAF0")
    TEXT         = colors.HexColor("#1A1033")
    MUTED        = colors.HexColor("#6B6580")
    BORDER       = colors.HexColor("#E5E0F5")
    BG           = colors.HexColor("#F7F4EF")
    WHITE        = colors.white

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "SmartTitle", parent=styles["Normal"],
        fontName="Helvetica-Bold", fontSize=22, textColor=WHITE,
        leading=28, alignment=TA_CENTER, spaceAfter=4,
    )
    subtitle_style = ParagraphStyle(
        "SmartSubtitle", parent=styles["Normal"],
        fontName="Helvetica", fontSize=12, textColor=colors.HexColor("#D8D0FF"),
        leading=16, alignment=TA_CENTER,
    )
    q_num_style = ParagraphStyle(
        "QNum", parent=styles["Normal"],
        fontName="Helvetica-Bold", fontSize=9, textColor=PURPLE,
        leading=12, spaceAfter=2,
    )
    q_text_style = ParagraphStyle(
        "QText", parent=styles["Normal"],
        fontName="Helvetica-Bold", fontSize=11, textColor=TEXT,
        leading=15,
    )
    a_label_style = ParagraphStyle(
        "ALabel", parent=styles["Normal"],
        fontName="Helvetica-Bold", fontSize=9, textColor=TEAL,
        leading=12, spaceAfter=2,
    )
    a_text_style = ParagraphStyle(
        "AText", parent=styles["Normal"],
        fontName="Helvetica", fontSize=10, textColor=TEXT,
        leading=14,
    )
    footer_style = ParagraphStyle(
        "Footer", parent=styles["Normal"],
        fontName="Helvetica", fontSize=8, textColor=MUTED,
        leading=12, alignment=TA_CENTER,
    )

    story = []

    # Cover header
    header_table = Table(
        [[Paragraph("📚 SmartTag Study Assistant", title_style)],
         [Paragraph(doc_name, subtitle_style)]],
        colWidths=[W - 40*mm],
    )
    header_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PURPLE),
        ("TOPPADDING", (0, 0), (-1, -1), 16),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 16),
        ("LEFTPADDING", (0, 0), (-1, -1), 20),
        ("RIGHTPADDING", (0, 0), (-1, -1), 20),
        ("ROUNDEDCORNERS", [8]),
    ]))
    story.extend([header_table, Spacer(1, 8*mm)])

    for idx, qa in enumerate(questions, 1):
        q_text = str(qa.get("q", "")).strip()
        a_text = str(qa.get("a", "")).strip()
        if not q_text:
            continue

        q_block = Table(
            [[Paragraph(f"Q{idx}", q_num_style)],
             [Paragraph(q_text, q_text_style)]],
            colWidths=[W - 40*mm],
        )
        q_block.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), PURPLE_LIGHT),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("LEFTPADDING", (0, 0), (-1, -1), 12),
            ("RIGHTPADDING", (0, 0), (-1, -1), 12),
            ("LINEBELOW", (0, -1), (-1, -1), 0.5, BORDER),
        ]))

        a_block = Table(
            [[Paragraph("ANSWER", a_label_style)],
             [Paragraph(a_text, a_text_style)]],
            colWidths=[W - 40*mm],
        )
        a_block.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), TEAL_LIGHT),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
            ("LEFTPADDING", (0, 0), (-1, -1), 12),
            ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ]))

        story.append(KeepTogether([q_block, a_block]))
        story.append(Spacer(1, 5*mm))

    story.append(HRFlowable(width="100%", thickness=0.5, color=BORDER))
    story.append(Spacer(1, 3*mm))
    story.append(Paragraph(
        f"SmartTag Study Assistant  •  Generated {datetime.now().strftime('%d %B %Y at %H:%M')}",
        footer_style
    ))

    doc.build(story)
    buf.seek(0)
    return buf


# ── Pydantic request models ───────────────────────────────────────────────────
class RegisterBody(BaseModel):
    email: str
    password: str
    name: str

class LoginBody(BaseModel):
    email: str
    password: str

class ChatBody(BaseModel):
    question: str
    doc_id: str | None = None
    session_id: str | None = None
    messages: list = []

class GenerateQuestionsBody(BaseModel):
    doc_id: str
    count: int = 10
    offset: int = 0

class ExportPDFBody(BaseModel):
    questions: list
    doc_name: str = "Study Material"


# ════════════════════════════════════════════════════════════════════════════
# AUTH ROUTES
# ════════════════════════════════════════════════════════════════════════════

@app.post("/api/auth/register", status_code=201)
async def register(body: RegisterBody, request: Request):
    email = body.email.strip().lower()
    password = body.password
    name = body.name.strip()
    if not email or not password or not name:
        raise HTTPException(400, "All fields required")
    if email in users:
        raise HTTPException(409, "Email already registered")
    if len(password) < 6:
        raise HTTPException(400, "Password min 6 chars")
    uid = str(uuid.uuid4())
    users[email] = {
        "id": uid, "name": name, "email": email,
        "password": hashlib.sha256(password.encode()).hexdigest(),
    }
    documents[uid] = []
    request.session.update({"user_id": uid, "user_name": name, "user_email": email})
    return {"message": "Registered", "user": {"name": name, "email": email}}


@app.post("/api/auth/login")
async def login(body: LoginBody, request: Request):
    email = body.email.strip().lower()
    user = users.get(email)
    if not user or user["password"] != hashlib.sha256(body.password.encode()).hexdigest():
        raise HTTPException(401, "Invalid email or password")
    request.session.update({"user_id": user["id"], "user_name": user["name"], "user_email": email})
    return {"message": "Login successful", "user": {"name": user["name"], "email": email}}


@app.post("/api/auth/logout")
async def logout(request: Request):
    request.session.clear()
    return {"message": "Logged out"}


@app.get("/api/auth/me")
async def me(request: Request):
    uid = request.session.get("user_id")
    if not uid:
        raise HTTPException(401, "Not authenticated")
    return {"user": {"name": request.session["user_name"], "email": request.session["user_email"]}}


# ════════════════════════════════════════════════════════════════════════════
# DOCUMENT ROUTES
# ════════════════════════════════════════════════════════════════════════════

@app.post("/api/documents/upload", status_code=201)
async def upload_document(request: Request, file: UploadFile = File(...)):
    uid = get_current_user(request)
    contents = await file.read()

    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File too large. Maximum upload size is 50 MB.")

    header = contents[:8]
    if not file.filename or not allowed_file(file.filename, header):
        raise HTTPException(400, "Unsupported file type. Allowed: PDF, TXT, DOC, DOCX, RTF, ODT")

    # Sanitise filename
    safe_name = re.sub(r"[^\w.\-]", "_", Path(file.filename).name)
    doc_id = str(uuid.uuid4())
    path = os.path.join(UPLOAD_FOLDER, doc_id + "_" + safe_name)

    with open(path, "wb") as f:
        f.write(contents)

    ext = safe_name.rsplit(".", 1)[-1].lower()
    pdf_type = "text"
    if ext == "pdf" and PDF_SUPPORT:
        try:
            reader = PdfReader(path)
            sample = "".join(
                (reader.pages[i].extract_text() or "")
                for i in range(min(2, len(reader.pages)))
            )
            if len(sample.strip()) < 100:
                pdf_type = "handwritten"
        except Exception:
            pdf_type = "handwritten"

    needs_ocr = pdf_type != "text"
    doc = {
        "id": doc_id,
        "name": safe_name,
        "size": len(contents),
        "uploaded_at": datetime.now().isoformat(),
        "path": path,
        "needs_ocr": needs_ocr,
        "pdf_type": pdf_type,
    }
    documents.setdefault(uid, []).append(doc)

    ocr_note = None
    if pdf_type == "handwritten":
        ocr_note = "Handwritten/image PDF detected — AI vision will be used to read it. Generation may take ~30s."

    return {"message": "Uploaded", "document": doc, "ocr_note": ocr_note}


@app.get("/api/documents")
async def get_documents(request: Request):
    uid = get_current_user(request)
    return {"documents": documents.get(uid, [])}


@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str, request: Request):
    uid = get_current_user(request)
    doc = next((d for d in documents.get(uid, []) if d["id"] == doc_id), None)
    if not doc:
        raise HTTPException(404, "Not found")
    try:
        os.remove(doc["path"])
    except Exception:
        pass
    documents[uid] = [d for d in documents[uid] if d["id"] != doc_id]
    _pdf_text_cache.pop(doc_id, None)
    return {"message": "Deleted"}


# ════════════════════════════════════════════════════════════════════════════
# CHAT ROUTE
# ════════════════════════════════════════════════════════════════════════════

@app.post("/api/chat")
async def chat(body: ChatBody, request: Request):
    uid = get_current_user(request)
    question = body.question.strip()
    if not question:
        raise HTTPException(400, "Question required")

    doc_id = body.doc_id
    session_id = body.session_id or str(uuid.uuid4())
    prior_messages = body.messages

    if doc_id:
        doc_name, doc_text = get_doc_context(doc_id, uid, max_chars=CHAT_CONTEXT_CHARS)
        if doc_text:
            system = (
                f'You are a Smart Study Assistant. The student uploaded "{doc_name}".\n\n'
                f'Read the student\'s question carefully and respond in the most appropriate format:\n\n'
                f'- LIST questions: reply ONLY with a clean numbered list.\n'
                f'- SHORT questions: give a clear direct answer in 2-4 sentences.\n'
                f'- EXPLAIN questions: give a thorough answer min 150 words.\n\n'
                f'DOCUMENT:\n{doc_text}'
            )
            source = "document"
        else:
            system = "You are a Smart Study Assistant. Answer in the format best suited to the question."
            source = "general"
    else:
        texts = []
        for doc in documents.get(uid, [])[:3]:
            t = _get_doc_full_text(doc)[:CHAT_CONTEXT_CHARS]
            if t:
                texts.append(f"[{doc['name']}]\n{t}")
        if texts:
            system = (
                "You are a Smart Study Assistant. Answer based on the uploaded documents.\n\n"
                + "DOCUMENTS:\n" + "\n\n---\n\n".join(texts)
            )
            source = "document"
        else:
            system = "You are a Smart Study Assistant."
            source = "general"

    messages = sanitise_messages(list(prior_messages)) + [{"role": "user", "content": question}]

    try:
        answer = groq_chat(system, messages, max_tokens=2048)
    except RateLimitError as e:
        raise HTTPException(429, detail={
            "error": "rate_limited", "retry_after": e.retry_after,
            "message": "Groq daily token limit reached. Please wait before sending another message."
        })
    except Exception as e:
        raise HTTPException(500, f"AI error: {str(e)}")

    msg = {
        "id": str(uuid.uuid4()),
        "question": question,
        "answer": answer,
        "doc_id": doc_id,
        "timestamp": datetime.now().isoformat(),
        "source": source,
    }
    chat_history.setdefault(session_id, []).append(msg)
    return {"message": msg, "session_id": session_id}


@app.get("/api/chat/history/{session_id}")
async def get_history(session_id: str, request: Request):
    get_current_user(request)
    return {"history": chat_history.get(session_id, [])}


# ════════════════════════════════════════════════════════════════════════════
# QUESTION GENERATION
# ════════════════════════════════════════════════════════════════════════════

@app.post("/api/generate-questions")
async def generate_questions(body: GenerateQuestionsBody, request: Request):
    uid = get_current_user(request)
    doc_id = body.doc_id
    count = min(body.count, 50)
    offset = body.offset
    task_id = str(uuid.uuid4())

    doc_name, doc_text = get_doc_context(doc_id, uid)
    if not doc_text:
        if not PDF_SUPPORT:
            raise HTTPException(400, "PDF library not installed. Run: pip install pypdf")
        raise HTTPException(400, "Could not extract text from this PDF.")

    CHUNK_SIZE = 3
    all_questions = []
    remaining = count
    batch_index = 0
    consecutive_fails = 0
    MAX_FAILS = 3
    doc_len = len(doc_text)
    # Divide the document into evenly-spaced windows so each batch
    # covers a different section — no wrap-around repetition
    total_windows = max(1, doc_len // DOC_CONTEXT_CHARS)

    try:
        while remaining > 0 and consecutive_fails < MAX_FAILS:
            if task_id in cancelled_tasks:
                cancelled_tasks.discard(task_id)
                if all_questions:
                    return {"questions": all_questions, "cancelled": True,
                            "warning": "Generation stopped by user."}
                raise HTTPException(499, "cancelled")

            batch = min(remaining, CHUNK_SIZE)

            # Slide through document sections; cycle only after full pass
            window_index = batch_index % total_windows
            context_start = offset + (window_index * DOC_CONTEXT_CHARS)
            context = doc_text[context_start:context_start + DOC_CONTEXT_CHARS]
            # If this window is empty (offset pushed past end), use last valid window
            if not context.strip():
                context = doc_text[max(0, doc_len - DOC_CONTEXT_CHARS):]

            already = len(all_questions)
            # Tell the AI exactly which questions already exist so it avoids repeating them
            existing_qs = ""
            if all_questions:
                existing_list = "\n".join(
                    f"- {q['q']}" for q in all_questions[-20:]  # last 20 to stay within token limit
                )
                existing_qs = f"\n\nALREADY GENERATED (DO NOT REPEAT ANY OF THESE):\n{existing_list}\n"

            batch_prompt = (
                f"Generate exactly {batch} exam-style questions from this document.\n"
                f"DOCUMENT: {doc_name}\nCONTENT:\n{context}\n"
                f"{existing_qs}"
                f"\nStart from question #{already + 1}. Every question MUST be completely UNIQUE "
                f"and DIFFERENT from all previously generated questions above.\n\n"
                f"Answer rules:\n"
                f"- Every answer MUST be at least 5-7 sentences long — never short.\n"
                f"- Structure each answer: (1) What it is, (2) How it works, "
                f"(3) Why it matters, (4) Key details or steps, (5) A real-world example, (6) Summary.\n"
                f"- Minimum 120 words per answer. Short answers are NOT acceptable.\n\n"
                f"IMPORTANT: You MUST return EXACTLY {batch} question objects. "
                f"Respond ONLY with a JSON array — no markdown, no explanation:\n"
                f'[{{"q":"Question?","a":"Detailed answer with minimum 120 words."}}]'
            )
            raw = groq_chat(
                system=(
                    "You are an expert exam question generator. "
                    "Respond ONLY with a valid JSON array. No markdown, no extra text. "
                    "Every answer MUST be at least 120 words and thoroughly detailed. "
                    "Never give short or one-line answers — always explain fully with examples."
                ),
                messages=[{"role": "user", "content": batch_prompt}],
                max_tokens=2500,
            )
            raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
            raw = re.sub(r"\s*```$", "", raw)

            batch_qs = None
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    parsed = next((v for v in parsed.values() if isinstance(v, list)), [])
                if isinstance(parsed, list) and len(parsed) > 0:
                    batch_qs = parsed
            except json.JSONDecodeError:
                m = re.search(r"\[.*\]", raw, re.DOTALL)
                if m:
                    try:
                        salvaged = json.loads(m.group())
                        if isinstance(salvaged, list) and len(salvaged) > 0:
                            batch_qs = salvaged
                    except json.JSONDecodeError:
                        pass

            if not batch_qs:
                consecutive_fails += 1
                batch_index += 1
                continue

            consecutive_fails = 0
            all_questions.extend(batch_qs)
            remaining -= len(batch_qs)
            batch_index += 1

    except RateLimitError as e:
        if all_questions:
            return {"questions": all_questions, "warning": "Partial results — daily token limit reached.",
                    "rate_limited": True, "retry_after": e.retry_after}
        raise HTTPException(429, detail={
            "error": "rate_limited", "retry_after": e.retry_after,
            "message": f"Groq daily token limit reached ({int(e.retry_after or 150)}s wait)."
        })
    except Exception as e:
        if all_questions:
            return {"questions": all_questions, "warning": "Partial results — server error mid-generation."}
        raise HTTPException(500, f"AI error: {str(e)}")
    finally:
        cancelled_tasks.discard(task_id)

    if not all_questions:
        raise HTTPException(500, "Could not generate questions, try again")

    return {"questions": all_questions[:count], "task_id": task_id}


# ════════════════════════════════════════════════════════════════════════════
# PDF EXPORT
# ════════════════════════════════════════════════════════════════════════════

@app.post("/api/export-pdf")
async def export_pdf(body: ExportPDFBody, request: Request):
    get_current_user(request)
    if not REPORTLAB_SUPPORT:
        raise HTTPException(500, "PDF export requires reportlab. Run: pip install reportlab")
    if not body.questions:
        raise HTTPException(400, "No questions provided")
    try:
        buf = build_styled_pdf(body.questions, body.doc_name)
        safe_name = re.sub(r"[^\w\-_\.]", "_", body.doc_name.replace(".pdf", ""))
        filename = f"SmartTag_{safe_name}_Questions.pdf"
        return StreamingResponse(
            buf,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    except Exception as e:
        raise HTTPException(500, f"PDF generation failed: {str(e)}")


# ════════════════════════════════════════════════════════════════════════════
# CANCELLATION
# ════════════════════════════════════════════════════════════════════════════

@app.post("/api/cancel/{task_id}")
async def cancel_task(task_id: str):
    cancelled_tasks.add(task_id)
    if len(cancelled_tasks) > 500:
        cancelled_tasks.clear()
    return {"cancelled": True}


# ════════════════════════════════════════════════════════════════════════════
# FRONTEND — serve static files
# ════════════════════════════════════════════════════════════════════════════

static_path = Path("static")
if static_path.exists():
    app.mount("/", StaticFiles(directory="static", html=True), name="static")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
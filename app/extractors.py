"""All file-text extraction logic with LRU caching."""

import os
import re
import io
import base64
import logging
from pathlib import Path

from PIL import Image as PILImage

from app.config import settings
from app.cache import pdf_text_cache

logger = logging.getLogger(__name__)

# ── Optional dependency detection ──────────────────────────────

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

# ── Lazy Groq client (avoids circular import) ─────────────────

_groq_client = None

def _get_groq_client():
    global _groq_client
    if _groq_client is None:
        from groq import Groq
        _groq_client = Groq(api_key=settings.GROQ_API_KEY)
    return _groq_client

# ── File validation ────────────────────────────────────────────

def allowed_file(filename: str, header: bytes | None = None) -> bool:
    if "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[1].lower()
    if ext not in settings.ALLOWED_EXTENSIONS:
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

# ── PDF extraction tiers ───────────────────────────────────────

def _pdf_to_images(path: str, dpi: int = 150, max_pages: int = 4) -> list:
    if not PDF_TO_IMAGE_SUPPORT:
        return []
    try:
        return convert_from_path(path, dpi=dpi, first_page=1, last_page=max_pages)
    except Exception as e:
        logger.error("PDF→image failed: %s", e)
        return []

def _image_to_b64(pil_img: PILImage.Image, max_width: int = 1024) -> str:
    w, h = pil_img.size
    if w > max_width:
        pil_img = pil_img.resize((max_width, int(h * max_width / w)), PILImage.LANCZOS)
    buf = io.BytesIO()
    pil_img.convert("RGB").save(buf, format="JPEG", quality=82)
    return base64.b64encode(buf.getvalue()).decode()

def _vision_extract(path: str, max_pages: int = 4) -> str:
    if not PDF_TO_IMAGE_SUPPORT:
        return ""
    try:
        images = _pdf_to_images(path, dpi=150, max_pages=max_pages)
        if not images:
            return ""
        client = _get_groq_client()
        parts: list[str] = []
        for i, img in enumerate(images):
            b64 = _image_to_b64(img)
            try:
                resp = client.chat.completions.create(
                    model=settings.GROQ_VISION_MODEL,
                    max_tokens=800,
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
        logger.error("Vision extract failed: %s", e)
        return ""

def _ocr_extract(path: str, max_chars: int = 6000) -> str:
    if not OCR_SUPPORT or not PDF_TO_IMAGE_SUPPORT:
        return ""
    try:
        images = _pdf_to_images(path, dpi=150, max_pages=4)
        parts: list[str] = []
        total = 0
        for i, img in enumerate(images):
            t = pytesseract.image_to_string(img, config="--oem 3 --psm 6")
            parts.append(f"[Page {i+1}]\n{t}")
            total += len(t)
            if total >= max_chars:
                break
        return "\n\n".join(parts)[:max_chars]
    except Exception as e:
        logger.error("OCR extract failed: %s", e)
        return ""

# ── Format-specific extractors ─────────────────────────────────

def _extract_txt(path: str, max_chars: int) -> str:
    try:
        with open(path, "r", errors="replace") as f:
            return f.read(max_chars)
    except Exception as e:
        logger.error("TXT extract failed: %s", e)
        return ""

def _extract_docx(path: str, max_chars: int) -> str:
    ext = path.rsplit(".", 1)[-1].lower()
    if ext == "docx":
        try:
            import docx as _docx
            doc = _docx.Document(path)
            return "\n".join(p.text for p in doc.paragraphs)[:max_chars]
        except Exception as e:
            logger.error("DOCX extract failed: %s", e)
    try:
        with open(path, "rb") as f:
            raw = f.read()
        chunks = re.findall(rb"[ -~]{4,}", raw)
        return "\n".join(c.decode("ascii", errors="ignore") for c in chunks)[:max_chars]
    except Exception as e:
        logger.error("DOC extract failed: %s", e)
        return ""

def _extract_rtf(path: str, max_chars: int) -> str:
    try:
        with open(path, "r", errors="replace") as f:
            raw = f.read()
        text = re.sub(r"\\[a-z]+[-\d]*[ ]?", " ", raw)
        text = re.sub(r"[{}\\]", "", text)
        return " ".join(text.split())[:max_chars]
    except Exception as e:
        logger.error("RTF extract failed: %s", e)
        return ""

def _extract_odt(path: str, max_chars: int) -> str:
    try:
        import zipfile
        import xml.etree.ElementTree as ET
        with zipfile.ZipFile(path) as z:
            with z.open("content.xml") as f:
                tree = ET.parse(f)
        parts = [el.text or "" for el in tree.iter() if el.text]
        return "\n".join(parts)[:max_chars]
    except Exception as e:
        logger.error("ODT extract failed: %s", e)
        return ""

# ── Public entry points ────────────────────────────────────────

def extract_pdf_text(path: str, max_chars: int = settings.PDF_MAX_CHARS) -> str:
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
            logger.error("Tier1 extract failed: %s", e)
            text = ""

    if len(text.strip()) >= 300:
        return text

    vision_text = _vision_extract(path, max_pages=4)
    if vision_text.strip():
        return vision_text[:max_chars]

    ocr_text = _ocr_extract(path, max_chars=max_chars)
    if ocr_text.strip():
        return ocr_text

    return ""

def extract_text_from_file(path: str, max_chars: int = settings.PDF_MAX_CHARS) -> str:
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

def get_doc_full_text(doc_path: str, doc_id: str) -> str:
    cached = pdf_text_cache.get(doc_id)
    if cached is not None:
        return cached
    text = extract_text_from_file(doc_path, max_chars=settings.DOC_CONTEXT_CHARS * 10)
    pdf_text_cache.set(doc_id, text)
    return text
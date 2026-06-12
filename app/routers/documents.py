"""Document upload, listing, and deletion routes."""

import os
import re
import uuid
import logging
import asyncio
from pathlib import Path
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.config import settings
from app.database import get_db, Document
from app.dependencies import get_current_user
from app.extractors import allowed_file
from app.cache import pdf_text_cache

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/documents", tags=["Documents"])

try:
    from pypdf import PdfReader
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False


# ── Helper to safely convert DB object to dict ────────────────
def _doc_to_dict(doc: Document) -> dict:
    """Safely convert SQLAlchemy Document to dict to avoid relationship serialization errors."""
    return {
        "id": doc.id,
        "name": doc.name,
        "size": doc.size,
        "path": doc.path,
        "needs_ocr": doc.needs_ocr,
        "pdf_type": doc.pdf_type,
        "uploaded_at": doc.uploaded_at.isoformat() if doc.uploaded_at else None,
    }


@router.post("/upload", status_code=201)
async def upload_document(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    uid = current_user["user_id"]
    contents = await file.read()

    if len(contents) > settings.MAX_UPLOAD_BYTES:
        raise HTTPException(413, detail={"error": "payload_too_large", "detail": "Max 50 MB"})

    header = contents[:8]
    if not file.filename or not allowed_file(file.filename, header):
        raise HTTPException(400, detail={"error": "invalid_file", "detail": "Unsupported file type"})

    safe_name = re.sub(r"[^\w.\-]", "_", Path(file.filename).name)
    doc_id = str(uuid.uuid4())
    path = os.path.join(settings.UPLOAD_FOLDER, doc_id + "_" + safe_name)

    # Write file to disk in a thread to avoid blocking
    await asyncio.to_thread(_write_file, path, contents)

    ext = safe_name.rsplit(".", 1)[-1].lower()
    pdf_type = "text"
    if ext == "pdf" and PDF_SUPPORT:
        try:
            # Run blocking PDF read in thread
            pdf_type = await asyncio.to_thread(_check_pdf_type, path)
        except Exception:
            pdf_type = "handwritten"

    needs_ocr = pdf_type != "text"
    doc = Document(
        id=doc_id, user_id=uid, name=safe_name, size=len(contents),
        path=path, needs_ocr=needs_ocr, pdf_type=pdf_type,
        uploaded_at=datetime.utcnow()
    )
    db.add(doc)
    await db.commit()

    ocr_note = None
    if pdf_type == "handwritten":
        ocr_note = "Handwritten/image PDF detected — AI vision will be used to read it."

    # Return the safe dictionary instead of the SQLAlchemy object
    return {"message": "Uploaded", "document": _doc_to_dict(doc), "ocr_note": ocr_note}


@router.get("")
async def get_documents(current_user: dict = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    uid = current_user["user_id"]
    result = await db.execute(select(Document).where(Document.user_id == uid))
    docs = result.scalars().all()
    
    # Convert all DB objects to safe dictionaries
    docs_list = [_doc_to_dict(doc) for doc in docs]
    return {"documents": docs_list}


@router.delete("/{doc_id}")
async def delete_document(doc_id: str, current_user: dict = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    uid = current_user["user_id"]
    result = await db.execute(select(Document).where(Document.id == doc_id, Document.user_id == uid))
    doc = result.scalars().first()
    
    if not doc:
        raise HTTPException(404, detail={"error": "not_found", "detail": "Document not found"})
    
    try:
        await asyncio.to_thread(os.remove, doc.path)
    except Exception:
        pass
    
    await db.delete(doc)
    await db.commit()
    pdf_text_cache.delete(doc_id)
    return {"message": "Deleted"}


# ── Helper Functions ───────────────────────────────────────────

def _write_file(path: str, contents: bytes):
    with open(path, "wb") as f:
        f.write(contents)

def _check_pdf_type(path: str) -> str:
    reader = PdfReader(path)
    sample = "".join((reader.pages[i].extract_text() or "") for i in range(min(2, len(reader.pages))))
    return "text" if len(sample.strip()) >= 100 else "handwritten"
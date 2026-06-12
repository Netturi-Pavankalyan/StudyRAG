"""PDF Export route."""

import re
import asyncio
import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from app.dependencies import get_current_user
from app.schemas import ExportPDFBody
from app.pdf_builder import build_styled_pdf, REPORTLAB_SUPPORT

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["Export"])


@router.post("/export-pdf")
async def export_pdf(body: ExportPDFBody, current_user: dict = Depends(get_current_user)):
    if not REPORTLAB_SUPPORT:
        raise HTTPException(500, detail={"error": "not_supported", "detail": "PDF export requires reportlab"})
    if not body.questions:
        raise HTTPException(400, detail={"error": "bad_request", "detail": "No questions provided"})
    
    try:
        # Run blocking PDF generation in a thread
        buf = await asyncio.to_thread(build_styled_pdf, body.questions, body.doc_name)
        safe_name = re.sub(r"[^\w\-_\.]", "_", body.doc_name.replace(".pdf", ""))
        filename = f"SmartTag_{safe_name}_Questions.pdf"
        
        return StreamingResponse(
            buf,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    except Exception as e:
        logger.error("PDF generation failed: %s", e)
        raise HTTPException(500, detail={"error": "export_failed", "detail": str(e)})
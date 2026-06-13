"""Chat with documents route."""

import uuid
import logging
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.config import settings
from app.database import get_db, Document, ChatSession, ChatMessage
from app.dependencies import get_current_user
from app.schemas import ChatBody
from app.extractors import get_doc_full_text
from app.groq_service import groq_chat, RateLimitError

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/chat", tags=["Chat"])

def sanitise_messages(messages: list[dict]) -> list[dict]:
    clean, total = [], 0
    for m in messages:
        role = m.get("role", "")
        content = str(m.get("content", ""))
        if role not in ("user", "assistant"):
            continue
        if len(content) > settings.CHAT_MESSAGE_MAX_CHARS:
            content = content[:settings.CHAT_MESSAGE_MAX_CHARS] + "…"
        total += len(content)
        if total > settings.CHAT_HISTORY_MAX_CHARS:
            break
        clean.append({"role": role, "content": content})
    return clean


@router.post("")
async def chat(body: ChatBody, current_user: dict = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    uid = current_user["user_id"]
    question = body.question.strip()
    if not question:
        raise HTTPException(400, detail={"error": "bad_request", "detail": "Question required"})

    session_id = body.session_id
    if session_id:
        result = await db.execute(select(ChatSession).where(ChatSession.id == session_id, ChatSession.user_id == uid))
        chat_session = result.scalars().first()
        if not chat_session:
            chat_session = ChatSession(id=session_id, user_id=uid, doc_id=body.doc_id)
            db.add(chat_session)
            await db.commit()
    else:
        session_id = str(uuid.uuid4())
        chat_session = ChatSession(id=session_id, user_id=uid, doc_id=body.doc_id)
        db.add(chat_session)
        await db.commit()

    doc_text = ""
    doc_name = ""
    source = "general"

    if body.doc_id:
        result = await db.execute(select(Document).where(Document.id == body.doc_id, Document.user_id == uid))
        doc = result.scalars().first()
        if doc:
            doc_name = doc.name
            doc_text = get_doc_full_text(doc.path, doc.id)
            source = "document"

    if not doc_text and not body.doc_id:
        result = await db.execute(select(Document).where(Document.user_id == uid).order_by(Document.uploaded_at.desc()).limit(3))
        docs = result.scalars().all()
        texts = []
        for doc in docs:
            t = get_doc_full_text(doc.path, doc.id)
            if t:
                texts.append(f"[{doc.name}]\n{t[:settings.CHAT_CONTEXT_CHARS]}")
        if texts:
            doc_text = "\n\n---\n\n".join(texts)
            source = "document"

    if doc_text:
        system = (
            f'You are a Smart Study Assistant. Student uploaded "{doc_name}".\n\n'
            f'=== STRICT ANSWER RULES ===\n\n'
            f'RULE 1 - For "topics" or "list" questions:\n'
            f'→ Give ONLY a clean numbered list of topic names.\n'
            f'→ NO page numbers, NO descriptions, NO definitions.\n'
            f'→ Example: 1. Fermat Theorem  2. RSA Algorithm  3. Modular Arithmetic\n\n'
            f'RULE 2 - For "explain X" questions:\n'
            f'Use this EXACT format:\n'
            f'[Topic Name]\n\n'
            f'Definition:\n'
            f'[definition]\n\n'
            f'Explanation:\n'
            f'[detailed explanation minimum 3 paragraphs]\n\n'
            f'Formula:\n'
            f'[formula if applicable, else skip]\n\n'
            f'Example:\n'
            f'[step by step example]\n\n'
            f'Key Points:\n'
            f'• [point 1]\n'
            f'• [point 2]\n'
            f'• [point 3]\n'
            f'• [point 4]\n'
            f'• [point 5]\n\n'
            f'RULE 3 - For "fill in the blanks" questions:\n'
            f'Use this EXACT format:\n'
            f'Fill in the Blanks:\n\n'
            f'1. _______ [sentence from document].\n'
            f'   Answer: [answer]\n\n'
            f'2. _______ [sentence].\n'
            f'   Answer: [answer]\n'
            f'(minimum 5 fill in the blanks)\n\n'
            f'RULE 4 - For "multiple choice" or "MCQ" questions:\n'
            f'Use this EXACT format:\n'
            f'Multiple Choice Questions:\n\n'
            f'1. [Question]?\n'
            f'   a) [wrong option]\n'
            f'   b) [correct option] ✅\n'
            f'   c) [wrong option]\n'
            f'   d) [wrong option]\n\n'
            f'2. [Question]?\n'
            f'   a) [option]\n'
            f'   b) [option]\n'
            f'   c) [correct option] ✅\n'
            f'   d) [option]\n'
            f'(minimum 5 MCQs)\n\n'
            f'RULE 5 - THREE CASES for all questions:\n'
            f'CASE 1: Topic fully in document → Answer completely from document.\n'
            f'CASE 2: Topic partially in document → Rewrite and COMPLETE using both\n'
            f'document content AND your knowledge. NO separate heading needed.\n'
            f'CASE 3: Topic NOT in document at all → Write EXACTLY:\n'
            f'**📚 Additional Information (Not in your document):**\n'
            f'Then give full answer below it.\n\n'
            f'RULE 6 - NEVER show page numbers in answers.\n'
            f'RULE 7 - NEVER give incomplete answers.\n\n'
            f'=== DOCUMENT CONTENT ===\n'
            f'Document: {doc_name}\n'
            f'{doc_text[:settings.CHAT_CONTEXT_CHARS]}'
        )
    else:
        system = (
            f'You are a Smart Study Assistant.\n'
            f'Answer EXACTLY what is asked, nothing more.\n'
            f'- "topics/list" → clean numbered list only, no page numbers\n'
            f'- "explain X" → Definition, Explanation, Formula, Example, Key Points\n'
            f'- "fill in the blanks" → Fill in the Blanks format with answers\n'
            f'- "MCQ/multiple choice" → Multiple Choice Questions with ✅ on correct answer\n'
            f'- "define" → definition only\n'
            f'- "summarize" → short summary'
        )

    messages = sanitise_messages(list(body.messages)) + [{"role": "user", "content": question}]

    try:
        answer = await groq_chat(system, messages, max_tokens=1024)
    except RateLimitError as e:
        raise HTTPException(429, detail={"error": "rate_limited", "retry_after": e.retry_after, "detail": "Groq daily token limit reached."})
    except Exception as e:
        raise HTTPException(500, detail={"error": "ai_error", "detail": str(e)})

    msg = ChatMessage(
        id=str(uuid.uuid4()), session_id=session_id, question=question,
        answer=answer, doc_id=body.doc_id, source=source, timestamp=datetime.utcnow()
    )
    db.add(msg)
    await db.commit()

    return {
        "message": {
            "id": msg.id, "question": msg.question, "answer": msg.answer,
            "doc_id": msg.doc_id, "timestamp": msg.timestamp.isoformat(), "source": msg.source
        },
        "session_id": session_id
    }


@router.get("/history/{session_id}")
async def get_history(session_id: str, current_user: dict = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(ChatMessage).where(ChatMessage.session_id == session_id).order_by(ChatMessage.timestamp)
    )
    messages = result.scalars().all()
    return {"history": messages}

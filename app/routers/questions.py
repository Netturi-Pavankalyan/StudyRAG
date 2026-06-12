"""Question generation and background task management."""

import uuid
import json
import re
import logging
import asyncio
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.config import settings
from app.database import get_db, Document, GenerationTask, async_session_factory
from app.dependencies import get_current_user
from app.schemas import GenerateQuestionsBody
from app.extractors import get_doc_full_text
from app.groq_service import groq_chat_sync, RateLimitError
from app.cache import cancelled_tasks
from app.dedup import filter_duplicate_questions

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["Questions"])


async def _run_generation_task(task_id: str, uid: str, doc_id: str, count: int, offset: int):
    """Background task that generates questions without blocking the API."""
    async with async_session_factory() as db:
        result = await db.execute(select(GenerationTask).where(GenerationTask.id == task_id))
        task = result.scalars().first()
        if not task:
            return
        
        task.status = "generating"
        await db.commit()

        result = await db.execute(select(Document).where(Document.id == doc_id, Document.user_id == uid))
        doc = result.scalars().first()
        if not doc:
            task.status = "failed"
            task.error = "Document not found"
            await db.commit()
            return

        doc_text = get_doc_full_text(doc.path, doc.id)
        if not doc_text:
            task.status = "failed"
            task.error = "Could not extract text"
            await db.commit()
            return

        all_questions = []
        remaining = count
        batch_index = 0
        consecutive_fails = 0
        doc_len = len(doc_text)
        total_windows = max(1, doc_len // settings.DOC_CONTEXT_CHARS)

        while remaining > 0 and consecutive_fails < 3:
            if task_id in cancelled_tasks:
                cancelled_tasks.discard(task_id)
                task.status = "cancelled"
                task.questions_json = json.dumps(all_questions)
                await db.commit()
                return

            batch = min(remaining, 8)
            window_index = batch_index % total_windows
            context_start = offset + (window_index * settings.DOC_CONTEXT_CHARS)
            context = doc_text[context_start:context_start + settings.DOC_CONTEXT_CHARS]
            if not context.strip():
                context = doc_text[max(0, doc_len - settings.DOC_CONTEXT_CHARS):]

            existing_qs = ""
            if all_questions:
                existing_list = "\n".join(f"- {q['q']}" for q in all_questions[-20:])
                existing_qs = f"\n\nALREADY GENERATED (DO NOT REPEAT):\n{existing_list}\n"

            batch_prompt = (
                f"Generate exactly {batch} exam-style questions from this document.\n"
                f"DOCUMENT: {doc.name}\nCONTENT:\n{context}\n"
                f"{existing_qs}"
                f"\nStart from question #{len(all_questions) + 1}.\n\n"
                f"Answer format - STRICTLY FOLLOW:\n"
                f"Line 1 - Definition: Two sentences explaining the concept clearly.\n"
                f"Line 2 - (blank line)\n"
                f"Line 3 - Key Points:\n"
                f"Line 4+ - minimum 5 bullet points starting with bullet symbol.\n\n"
                f"IMPORTANT: Return EXACTLY {batch} question objects. "
                f"Respond ONLY with a JSON array:\n"
                f'[{{"q":"Question?","a":"Definition: Sentence one. Sentence two.\\n\\nKey Points:\\n• Point 1\\n• Point 2\\n• Point 3\\n• Point 4\\n• Point 5\\n• Point 6"}}]'
            )
            
            try:
                # Run sync Groq call in a thread to not block the async loop
                raw = await asyncio.to_thread(
                    groq_chat_sync,
                    system="You are an expert exam question generator. Respond ONLY with a valid JSON array. No markdown.",
                    messages=[{"role": "user", "content": batch_prompt}],
                    max_tokens=2000
                )
                
                logger.info("RAW GROQ RESPONSE: %s", raw[:500])
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
                            batch_qs = json.loads(m.group())
                        except json.JSONDecodeError:
                            pass

                if not batch_qs:
                    consecutive_fails += 1
                    batch_index += 1
                    continue

                # Deduplicate
                unique_batch = filter_duplicate_questions(all_questions, batch_qs)
                all_questions.extend(unique_batch)
                remaining -= len(unique_batch)
                consecutive_fails = 0
                
                # Update progress
                progress = int((count - remaining) / count * 100)
                task.progress = progress
                task.questions_json = json.dumps(all_questions)
                await db.commit()

            except RateLimitError:
                task.status = "failed"
                task.error = "Rate limited by Groq"
                task.questions_json = json.dumps(all_questions) # Save partials
                await db.commit()
                return
            except Exception as e:
                logger.error("Generation batch error: %s", e)
                consecutive_fails += 1

            batch_index += 1

        task.status = "completed"
        task.progress = 100
        task.questions_json = json.dumps(all_questions[:count])
        task.updated_at = datetime.utcnow()
        await db.commit()


@router.post("/generate-questions")
async def generate_questions(
    body: GenerateQuestionsBody,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    uid = current_user["user_id"]
    count = min(body.count, 50)
    
    # Ensure doc exists and belongs to user
    result = await db.execute(select(Document).where(Document.id == body.doc_id, Document.user_id == uid))
    doc = result.scalars().first()
    if not doc:
        raise HTTPException(404, detail={"error": "not_found", "detail": "Document not found"})

    # Cancel any existing running tasks for this user+doc
    from sqlalchemy import update
    await db.execute(
        update(GenerationTask)
        .where(GenerationTask.user_id == uid, 
               GenerationTask.doc_id == body.doc_id,
               GenerationTask.status.in_(["pending", "generating"]))
        .values(status="cancelled")
    )
    await db.commit()

    task = GenerationTask(
        id=str(uuid.uuid4()), user_id=uid, doc_id=body.doc_id,
        status="pending", total_requested=count, progress=0
    )
    db.add(task)
    await db.commit()
    await db.refresh(task)

    # Spawn background task
    asyncio.create_task(_run_generation_task(task.id, uid, body.doc_id, count, body.offset))

    return {"message": "Generation started", "task_id": task.id}


@router.get("/generate-questions/progress/{task_id}")
async def get_generation_progress(task_id: str, current_user: dict = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    uid = current_user["user_id"]
    result = await db.execute(select(GenerationTask).where(GenerationTask.id == task_id, GenerationTask.user_id == uid))
    task = result.scalars().first()
    
    if not task:
        raise HTTPException(404, detail={"error": "not_found", "detail": "Task not found"})

    return {
        "task_id": task.id,
        "status": task.status,
        "progress": task.progress,
        "questions": json.loads(task.questions_json) if task.status in ["completed", "cancelled"] else [],
        "error": task.error
    }


@router.post("/cancel/{task_id}")
async def cancel_task(task_id: str, current_user: dict = Depends(get_current_user)):
    # Auth-enforced cancellation
    cancelled_tasks.add(task_id)
    return {"cancelled": True}
"""Deduplication logic for generated questions using string similarity."""

import logging
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)

def is_similar(q1: str, q2: str, threshold: float = 0.75) -> bool:
    """Returns True if two strings are > threshold similar."""
    return SequenceMatcher(None, q1.lower().strip(), q2.lower().strip()).ratio() > threshold

def filter_duplicate_questions(
    existing_questions: list[dict], 
    new_questions: list[dict], 
    threshold: float = 0.75
) -> list[dict]:
    """
    Removes new questions that are too similar to existing ones.
    """
    existing_texts = [q.get("q", "") for q in existing_questions]
    unique_new = []
    
    for nq in new_questions:
        nq_text = nq.get("q", "")
        if not nq_text:
            continue
            
        is_dup = False
        for eq_text in existing_texts:
            if is_similar(nq_text, eq_text, threshold):
                is_dup = True
                logger.debug("Filtered duplicate question: %s", nq_text[:50])
                break
                
        if not is_dup:
            unique_new.append(nq)
            existing_texts.append(nq_text) # Prevent duplicates within the new batch itself
            
    return unique_new
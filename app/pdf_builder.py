"""Styled PDF builder for question exports."""

import io
import logging
from datetime import datetime

from app.config import settings

logger = logging.getLogger(__name__)

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


def build_styled_pdf(questions: list[dict], doc_name: str) -> io.BytesIO:
    if not REPORTLAB_SUPPORT:
        raise ImportError("PDF export requires reportlab. Run: pip install reportlab")

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=20*mm, rightMargin=20*mm,
        topMargin=22*mm, bottomMargin=22*mm,
    )
    W, H = A4

    PURPLE       = colors.HexColor("#6C4FF6")
    PURPLE_LIGHT = colors.HexColor("#EDE9FE")
    TEAL         = colors.HexColor("#0DB884")
    TEAL_LIGHT   = colors.HexColor("#D1FAF0")
    TEXT         = colors.HexColor("#1A1033")
    MUTED        = colors.HexColor("#6B6580")
    BORDER       = colors.HexColor("#E5E0F5")
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
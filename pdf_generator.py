"""PDF report generation module.

Builds a downloadable PDF incident report from the AI analysis text
using the fpdf2 library.
"""

from __future__ import annotations

from datetime import datetime, timezone

from fpdf import FPDF


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_TITLE_FONT_SIZE = 18
_HEADING_FONT_SIZE = 13
_BODY_FONT_SIZE = 11
_MARGIN = 15
_LINE_HEIGHT = 7


def _add_title(pdf: FPDF, title: str) -> None:
    """Add the report title and generation timestamp."""
    pdf.set_font("Helvetica", style="B", size=_TITLE_FONT_SIZE)
    pdf.set_text_color(30, 30, 120)
    pdf.cell(0, 12, title, new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_text_color(80, 80, 80)
    pdf.set_font("Helvetica", size=9)
    timestamp = datetime.now(timezone.utc).strftime("%d %B %Y, %H:%M UTC")
    pdf.cell(0, 6, f"Generated: {timestamp}", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(4)


def _add_section_heading(pdf: FPDF, text: str) -> None:
    """Render a coloured section heading."""
    pdf.set_font("Helvetica", style="B", size=_HEADING_FONT_SIZE)
    pdf.set_text_color(30, 30, 120)
    pdf.cell(0, 9, text, new_x="LMARGIN", new_y="NEXT")
    pdf.set_draw_color(30, 30, 120)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
    pdf.ln(3)


def _add_body_text(pdf: FPDF, text: str) -> None:
    """Render multi-line body text."""
    pdf.set_font("Helvetica", size=_BODY_FONT_SIZE)
    pdf.set_text_color(40, 40, 40)
    pdf.multi_cell(0, _LINE_HEIGHT, text)
    pdf.ln(3)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_report(
    analysis_text: str,
    safety_rules: str,
    video_filename: str,
) -> bytes:
    """Create a PDF incident report and return it as raw bytes.

    Args:
        analysis_text:  Raw text returned by the Gemini model.
        safety_rules:   The safety-rules text supplied by the user.
        video_filename: Original name of the uploaded video file.

    Returns:
        The PDF document as a ``bytes`` object, ready for download.
    """
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_margins(_MARGIN, _MARGIN, _MARGIN)
    pdf.set_auto_page_break(auto=True, margin=_MARGIN)
    pdf.add_page()

    _add_title(pdf, "Airside Safety Incident Report")

    _add_section_heading(pdf, "Video Analysed")
    _add_body_text(pdf, video_filename)

    _add_section_heading(pdf, "Safety Rules Applied")
    _add_body_text(pdf, safety_rules)

    _add_section_heading(pdf, "AI Analysis - Detected Violations")
    _add_body_text(pdf, analysis_text)

    # fpdf2 output() with dest="S" returns a bytearray; cast to bytes
    return bytes(pdf.output())

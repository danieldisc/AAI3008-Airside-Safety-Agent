import re
from fpdf import FPDF
from datetime import datetime
 
def clean_text_for_pdf(text):
    """
    Sanitizes text and aggressively breaks up long filenames/strings
    so FPDF doesn't crash on word wraps.
    """
    if not isinstance(text, str):
        text = str(text)
        
    replacements = {
        "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
        "\u2013": "-", "\u2014": "-", "\u2026": "...",
        "\u26a0\ufe0f": "", "\u2705": "", "\u274c": "", "\U0001f6e1\ufe0f": "",
        "\U0001f7e2": "", "\U0001f7e1": "", "\U0001f534": "",
        "\U0001f6e0\ufe0f": "", "\U0001f4d6": "",
        "_": " "  # Replace underscores so filenames wrap gracefully
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
        
    text = text.encode('latin-1', 'replace').decode('latin-1')
    
    # ULTIMATE FALLBACK: break any continuous run of 45+ non-space chars
    text = re.sub(r'([^\s]{45})', r'\1 ', text)
    
    return text
 
 
def _safe_multi_cell(pdf, h, text):
    """
    Helper that always resets X to the left margin before calling multi_cell,
    and uses the full printable width — preventing the
    'Not enough horizontal space to render a single character' crash.
    """
    pdf.set_x(pdf.l_margin)
    printable_w = pdf.w - pdf.l_margin - pdf.r_margin
    pdf.multi_cell(printable_w, h, text)
 
 
def create_pdf_report(report_data, filename="incident_report.pdf"):
    """
    Generates a structured PDF from the deterministic RAG JSON payload.
    """
    class PDF(FPDF):
        def header(self):
            self.set_font('Arial', 'B', 15)
            self.set_x(self.l_margin)
            self.cell(0, 10, 'AIRSIDE SAFETY AUDIT REPORT', 0, 1, 'C')
            self.ln(5)
 
        def footer(self):
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
 
    pdf = PDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
 
    # ── 1. Metadata & Risk Summary ──────────────────────────────────────────
    summary = report_data.get('summary', {})
    risk = summary.get('overall_risk', 'UNKNOWN').upper()
 
    pdf.set_font("Arial", 'B', 11)
    pdf.cell(0, 8, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
    pdf.cell(0, 8, clean_text_for_pdf(f"Clip ID: {report_data.get('clip_id', 'Unknown')}"), ln=True)
 
    if risk == "HIGH":
        pdf.set_text_color(200, 0, 0)
    elif risk == "MEDIUM":
        pdf.set_text_color(200, 150, 0)
    else:
        pdf.set_text_color(0, 150, 0)
 
    pdf.cell(0, 10, f"OVERALL RISK LEVEL: {risk}", ln=True)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(5)
 
    # ── 2. Top Findings & Citations ─────────────────────────────────────────
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "TOP SAFETY FINDINGS", ln=True)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.l_margin + 190, pdf.get_y())
    pdf.ln(3)
 
    findings = summary.get('top_findings', [])
    if not findings:
        pdf.set_font("Arial", '', 11)
        pdf.cell(0, 8, "No safety violations detected.", ln=True)
    else:
        for finding in findings:
            # Violation header — multi_cell so long labels wrap safely
            pdf.set_font("Arial", 'B', 11)
            severity = (finding.get('severity') or '').upper()
            _safe_multi_cell(pdf, 6, clean_text_for_pdf(
                f"Violation: {finding.get('label')} ({severity})"
            ))
 
            # Code / confidence on one line (short, safe to use cell)
            pdf.set_font("Arial", '', 10)
            pdf.set_x(pdf.l_margin)
            pdf.cell(0, 6, clean_text_for_pdf(
                f"Code: {finding.get('violation_code')} | Confidence: {finding.get('confidence')}"
            ), ln=True)
 
            # SOP citations
            for claim in report_data.get('claims', []):
                for match in claim.get('matched_violations', []):
                    if match.get('violation_code') == finding.get('violation_code'):
                        for citation in match.get('citations', []):
                            pdf.set_font("Arial", 'I', 10)
                            _safe_multi_cell(pdf, 6, clean_text_for_pdf(
                                f"  Source SOP: {citation}"
                            ))
            pdf.ln(4)
 
    # ── 3. Recommended Actions ───────────────────────────────────────────────
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "RECOMMENDED IMMEDIATE ACTIONS", ln=True)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.l_margin + 190, pdf.get_y())
    pdf.ln(3)
 
    actions = report_data.get('recommended_immediate_actions', [])
    pdf.set_font("Arial", '', 11)
    if not actions:
        pdf.cell(0, 8, "None required.", ln=True)
    else:
        for action in actions:
            _safe_multi_cell(pdf, 6, clean_text_for_pdf(f"- {action}"))
            pdf.ln(2)
 
    pdf.output(filename)
    return filename
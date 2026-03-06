from fpdf import FPDF
from datetime import datetime

def clean_text_for_pdf(text):
    """
    Removes unsupported characters that crash the PDF generator.
    """
    replacements = {
        "’": "'", "‘": "'", "“": '"', "”": '"',  # Smart quotes
        "–": "-", "—": "-", "…": "...",          # Dashes/Ellipses
        "⚠️": "", "✅": "", "❌": "", "🛡️": "",    # Emojis (just in case)
        "\u2019": "'"                            # Unicode specific apostrophe
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Final safety net: Remove anything that isn't standard Latin-1 text
    return text.encode('latin-1', 'replace').decode('latin-1')

def create_pdf_report(report_text, filename="incident_report.pdf"):
    class PDF(FPDF):
        def header(self):
            self.set_font('Arial', 'B', 15)
            self.cell(0, 10, 'AIRSIDE SAFETY AUDIT REPORT', 0, 1, 'C')
            self.ln(5)

        def footer(self):
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    pdf = PDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Metadata
    pdf.set_font("Arial", size=10)
    pdf.cell(0, 10, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.cell(0, 10, "System: Gemini 1.5 Flash Vision / AAI3008 Agent", ln=True)
    pdf.ln(5)
    
    # Clean the text!
    safe_text = clean_text_for_pdf(report_text)
    
    # Body
    pdf.set_font("Arial", size=11)
    
    # Basic Markdown Cleanup (Bold/Lists)
    safe_text = safe_text.replace("**", "").replace("##", "").replace("* ", "- ")
    
    pdf.multi_cell(0, 6, safe_text)
    
    pdf.output(filename)
    return filename
"""Airside Safety Agent – main Streamlit application.

Usage:
    streamlit run app.py

The app accepts an MP4 video and a set of safety rules, sends both to the
Google Gemini 1.5 Flash multimodal model for analysis, then generates a
downloadable PDF incident report from the response.
"""

import os
import tempfile

import streamlit as st

from gemini_client import analyse_video
from pdf_generator import generate_report

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Airside Safety Agent",
    page_icon="✈️",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("✈️ Airside Safety Agent")
st.markdown(
    """
    Upload an airside video and enter the applicable safety rules.
    The AI agent will analyse the footage for violations and produce
    a downloadable PDF incident report.
    """
)

st.divider()

# ---------------------------------------------------------------------------
# Sidebar – API key
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input(
        "Google Gemini API Key",
        type="password",
        help="Your Google Gemini API key. It is never stored or logged.",
    )
    st.caption(
        "Obtain a free key at "
        "[Google AI Studio](https://aistudio.google.com/app/apikey)."
    )

# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

st.subheader("1. Upload Video")
uploaded_video = st.file_uploader(
    "Drag and drop an MP4 video file here",
    type=["mp4"],
    help="Supported format: MP4. Maximum size is determined by your Gemini plan.",
)

st.subheader("2. Enter Safety Rules")
default_rules = (
    "1. No personnel within 10 metres of an operating engine.\n"
    "2. All ground crew must wear high-visibility vests.\n"
    "3. Ground support equipment must not exceed 15 km/h on the apron.\n"
    "4. Aircraft must be chocked before ground support equipment approaches.\n"
    "5. No smoking within 50 metres of any aircraft."
)
safety_rules = st.text_area(
    "Safety rules (one per line or free text)",
    value=default_rules,
    height=160,
    help="List the rules that the AI should check against.",
)

# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

st.divider()

run_analysis = st.button(
    "🔍 Analyse Video",
    type="primary",
    disabled=(uploaded_video is None or not api_key.strip()),
)

if uploaded_video is not None and not api_key.strip():
    st.info("Please enter your Gemini API key in the sidebar to run the analysis.")

if run_analysis:
    with st.spinner("Uploading video and analysing with Gemini 1.5 Flash…"):
        # Write the uploaded bytes to a temporary file so the Gemini SDK can
        # access it by file path.
        suffix = os.path.splitext(uploaded_video.name)[1] or ".mp4"
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_video.read())
                tmp_path = tmp.name

            analysis_text = analyse_video(
                api_key=api_key.strip(),
                video_path=tmp_path,
                safety_rules=safety_rules,
            )
        except Exception as exc:  # noqa: BLE001
            st.error(f"Analysis failed: {exc}")
            st.stop()
        finally:
            # Remove the temporary file regardless of success or failure
            try:
                os.unlink(tmp_path)
            except Exception:  # noqa: BLE001
                pass

    st.success("Analysis complete!")

    st.subheader("📋 Detected Violations")
    st.markdown(analysis_text)

    st.divider()

    # -----------------------------------------------------------------------
    # PDF generation
    # -----------------------------------------------------------------------

    st.subheader("📄 Download Incident Report")
    with st.spinner("Generating PDF report…"):
        try:
            pdf_bytes = generate_report(
                analysis_text=analysis_text,
                safety_rules=safety_rules,
                video_filename=uploaded_video.name,
            )
        except Exception as exc:  # noqa: BLE001
            st.error(f"PDF generation failed: {exc}")
            st.stop()

    st.download_button(
        label="⬇️ Download PDF Report",
        data=pdf_bytes,
        file_name="airside_safety_incident_report.pdf",
        mime="application/pdf",
    )

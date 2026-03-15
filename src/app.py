import streamlit as st
from pathlib import Path
import os
import time
from vlm_agent import SafetyAgent
from report_gen import create_pdf_report, extract_section
from evaluate import evaluate_observer_phase, evaluate_analyst_phase

# RAG Pipeline Imports
from rag.vlm_incident import _build_incident_payload
from rag.config import INDEX_DIR, DEFAULT_TOP_K
from rag.incident_retrieval import retrieve_for_incident
from rag.llm2_mapper import map_retrieval_payload, DEFAULT_RULE_PACK_DIR
from rag.llm3_teachable import build_coaching_payload
from rag.llm4_report import build_report

st.set_page_config(
    page_title="Changi Airside Safety Audit",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CACHE THE AGENT ---
# This prevents the app from reloading the API keys every single time you click a button!
@st.cache_resource
def get_agent():
    return SafetyAgent()

try:
    agent = get_agent()
except Exception as e:
    st.error(f"Configuration Error: {e}. Please check your API keys.")
    st.stop()

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; color: #e0e0e0; }
    .stApp { background-color: #0f1116; }
    .report-container { background-color: #161b22; padding: 24px; border-radius: 6px; border: 1px solid #30363d; margin-top: 20px; }
    h1, h2, h3 { color: #ffffff; font-weight: 600; letter-spacing: -0.5px; }
    .stButton>button { border-radius: 6px; font-weight: 500; height: 45px; border: 1px solid #30363d; }
    .stButton>button:hover { border-color: #8b949e; color: #ffffff; }
    section[data-testid="stSidebar"] { background-color: #0d1117; border-right: 1px solid #30363d; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.markdown("### Configuration")
    
    # DANIEL'S UPGRADE: The Dual-Engine Toggle!
    st.markdown("---")
    st.markdown("**AI Intelligence Engine**")
    engine_choice = st.radio("Select Engine:", ["Gemini", "OpenAI"])
    
    if engine_choice == "OpenAI" and not agent.openai_client:
        st.error("OpenAI API Key missing! Add it to .env or secrets.toml")
        
    st.markdown("---")
    st.markdown("**Enforcement Protocols**")
    rule_exclude = st.checkbox("Exclusion Zones (3m)", value=True)
    
    st.markdown("---")
    # DANIEL'S UPGRADE: Quota Saver Button
    if st.button("Clear Cache & Reset"):
        for key in ['analysis_result', 'full_logs', 'file_processed']:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()
        
    st.caption("System Version: v3.1 (Dual-Engine Map-Reduce)")

# --- MAIN UI ---
st.title("Automated Incident Investigation")
st.markdown("Upload CCTV footage (up to 5 mins). The system chunks the video and utilises an Observer AI to monitor states.")

uploaded_file = st.file_uploader("Source Footage (MP4/MOV)", type=["mp4", "mov"])

if uploaded_file is not None:
    st.video(uploaded_file)
    
    if st.button(f"Run Compliance Analysis via {engine_choice}", type="primary"):
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        progress_bar = st.progress(0, text="Initialising Pipeline...")
        
        def update_progress(percent, text):
            progress_bar.progress(percent, text=text)
            
        try:
            full_logs, analysis_text, processing_time = agent.analyze_pipeline(
                temp_path, 
                progress_callback=update_progress,
                engine=engine_choice 
            )
            
            update_progress(85, text="Structuring data and querying SOP manuals...")
            
            incident_payload = _build_incident_payload(
                video_path=Path(temp_path), 
                clip_id=uploaded_file.name, 
                logs=full_logs, 
                analyst_text=analysis_text
            )
            
            # 3. Run the Deterministic RAG Pipeline
            retrieval_payload = retrieve_for_incident(
                incident_payload, index_dir=INDEX_DIR, top_k=DEFAULT_TOP_K
            )
            violations_payload = map_retrieval_payload(
                retrieval_payload, rule_pack_dir=DEFAULT_RULE_PACK_DIR
            )
            teachable_payload = build_coaching_payload(violations_payload)
            final_report = build_report(
                incident_payload, retrieval_payload, violations_payload, teachable_payload
            )
            
            update_progress(100, text="Pipeline Execution Complete.")
            time.sleep(1)
            progress_bar.empty()
            
            # Save to session state
            st.session_state['analysis_result'] = analysis_text
            st.session_state['full_logs'] = full_logs
            st.session_state['structured_report'] = final_report  # New structured payload
            st.session_state['file_processed'] = True
            st.session_state['processing_time'] = processing_time

        except Exception as e:
            st.error(f"Processing Error: {e}")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

if st.session_state.get('file_processed') and 'structured_report' in st.session_state:
    report = st.session_state['structured_report']
    summary = report.get('summary', {})
    
    st.markdown("---")
    
    # Dynamic Risk Header
    risk = summary.get('overall_risk', 'unknown').lower()
    risk_color = "🔴" if risk == "high" else "🟡" if risk == "medium" else "🟢"
    st.subheader(f"Incident Report ({engine_choice}) - Risk Level: {risk_color} {risk.upper()}")
    
    # 1. Top Findings Section
    st.markdown("### Top Safety Findings")
    findings = summary.get('top_findings', [])
    if not findings:
        st.success("✅ No safety violations detected based on standard operating procedures.")
    else:
        for finding in findings:
            with st.expander(f"⚠️ {finding.get('label')} ({finding.get('severity', '').upper()})", expanded=True):
                st.write(f"**Violation Code:** `{finding.get('violation_code')}`")
                st.write(f"**Confidence:** {finding.get('confidence')}")
                
                # Dig into the report to find the specific citations for this violation
                for claim in report.get('claims', []):
                    for match in claim.get('matched_violations', []):
                        if match.get('violation_code') == finding.get('violation_code'):
                            citations = match.get('citations', [])
                            if citations:
                                st.markdown("**SOP Citations:**")
                                for citation in citations:
                                    st.markdown(f"- 📖 *{citation}*")
                                    
    # 2. Recommended Actions Section
    actions = report.get('recommended_immediate_actions', [])
    if actions:
        st.markdown("### Recommended Immediate Actions")
        for action in actions:
            st.markdown(f"- 🛠️ {action}")

    st.markdown("<br>", unsafe_allow_html=True)
    
    # 3. VLM Narrative Breakdown
    raw_narrative = st.session_state['analysis_result']
    vlm_summary = extract_section(raw_narrative, "OVERALL INCIDENT SUMMARY")
    vlm_sequence = extract_section(raw_narrative, "COMBINED NARRATIVE SEQUENCE")
    vlm_root_cause = extract_section(raw_narrative, "COMPREHENSIVE ROOT CAUSE OBSERVATION")

    with st.expander("📝 View Detailed Incident Narrative", expanded=True):
        if not vlm_summary and not vlm_sequence:
            # Fallback just in case the VLM forgot its formatting headers
            st.write(raw_narrative)
        else:
            if vlm_summary:
                st.markdown("**OVERALL INCIDENT SUMMARY**")
                st.write(vlm_summary)
            if vlm_sequence:
                st.markdown("**COMBINED NARRATIVE SEQUENCE**")
                st.write(vlm_sequence)
            if vlm_root_cause:
                st.markdown("**COMPREHENSIVE ROOT CAUSE OBSERVATION**")
                st.write(vlm_root_cause)
        
    with st.expander("🔎 View Raw Frame-by-Frame Observer Logs"):
        st.json(st.session_state['full_logs'])
    
    # 4. Export PDF (Using our updated report_gen.py)
    pdf_filename = "safety_report.pdf"
    create_pdf_report(report, st.session_state['analysis_result'], pdf_filename)
    
    with open(pdf_filename, "rb") as pdf_file:
        st.download_button(
            label="Export Structured PDF Report",
            data=pdf_file,
            file_name="Preliminary_Incident_Report.pdf",
            mime="application/pdf"
        )
        
    # --- Automated Evaluation Section ---
    st.markdown("---")
    st.subheader("📊 Automated Evaluation Metrics")
    
    if 'processing_time' in st.session_state:
        st.metric("⏱️ VLM Pipeline Processing Time", f"{st.session_state['processing_time']} seconds")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_name = os.path.splitext(uploaded_file.name)[0]
    eval_folder = os.path.join(current_dir, "eval_data", base_name)
    
    truth_json_path = os.path.join(eval_folder, f"{base_name}_truths.json")
    truth_report_path = os.path.join(eval_folder, f"{base_name}_report.txt")
    
    if os.path.exists(truth_json_path) or os.path.exists(truth_report_path):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Phase 1: Observer (Detection)**")
            if os.path.exists(truth_json_path):
                obs_metrics = evaluate_observer_phase(
                    truth_json_path, 
                    st.session_state['full_logs'], 
                    chunk_size=10
                )
                
                st.metric("Chunk Accuracy", f"{obs_metrics['accuracy'] * 100:.1f}%")
                st.metric("Precision", f"{obs_metrics['precision'] * 100:.1f}%")
                st.metric("Recall", f"{obs_metrics['recall'] * 100:.1f}%")
                st.metric("F1 Score", f"{obs_metrics['f1_score'] * 100:.1f}%")
            else:
                st.info(f"Missing `{base_name}_truths.json` for Phase 1.")
                
        with col2:
            st.markdown("**Phase 2: Analyst (Narrative)**")
            if os.path.exists(truth_report_path):
                with st.spinner("Calculating NLP Metrics (BERTScore & METEOR)..."):
                    ana_metrics = evaluate_analyst_phase(
                        truth_report_path, 
                        st.session_state['analysis_result']
                    )
                    
                st.metric("BERTScore (F1)", f"{ana_metrics['bert_f1']:.3f}")
                st.metric("METEOR Score", f"{ana_metrics['meteor_score']:.3f}")
            else:
                st.info(f"Missing `{base_name}_report.txt` for Phase 2.")
                
    else:
        st.info(f"No ground truth data found for `{base_name}` in the `eval_data` folder. Upload a benchmark video to see evaluation metrics.")
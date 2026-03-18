import streamlit as st
import threading
from streamlit.runtime.scriptrunner import add_script_run_ctx
from pathlib import Path
import os
import time
import json
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
    initial_sidebar_state="collapsed"
)

# --- CACHE THE AGENT ---
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
    h1, h2, h3 { color: #ffffff; font-weight: 600; letter-spacing: -0.5px; }
    .stButton>button { border-radius: 6px; font-weight: 500; height: 45px; border: 1px solid #30363d; }
    .stButton>button:hover { border-color: #8b949e; color: #ffffff; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# --- HEADER SECTION ---
header_col1, header_col2 = st.columns([4, 1])
with header_col1:
    st.title("Automated Incident Investigation")
    st.markdown("Upload CCTV footage (up to 5 mins). The system runs a **Dual-Engine Map-Reduce** evaluation using both Gemini and OpenAI simultaneously.")
with header_col2:
    st.write("") # Spacing
    if st.button("🔄 Clear Cache & Reset", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# --- MAIN UI ---
uploaded_file = st.file_uploader("Source Footage (MP4/MOV)", type=["mp4", "mov"])

if uploaded_file is not None:
    # Display the video centered
    _, vid_col, _ = st.columns([1, 2, 1])
    with vid_col:
        st.video(uploaded_file)
    
    if st.button("Run Dual-Engine Compliance Analysis", type="primary", use_container_width=True):
        import shutil
        base_name = os.path.splitext(uploaded_file.name)[0]
        eval_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_data", base_name)
        os.makedirs(eval_data_dir, exist_ok=True)

        # Save uploaded video to eval_data_dir
        video_path = os.path.join(eval_data_dir, uploaded_file.name)
        with open(video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        temp_path = video_path  # Use the saved path for processing
        st.session_state['file_processed'] = False

        st.markdown("---")
        exec_col1, exec_col2 = st.columns(2)

        with exec_col1:
            st.markdown("### 🔵 Gemini Pipeline")
            prog_g = st.progress(0, text="Initialising...")

        with exec_col2:
            st.markdown("### 🟢 OpenAI Pipeline")
            prog_o = st.progress(0, text="Initialising...")

        # --- DEFINE THREAD FUNCTIONS ---
        def run_gemini():
            try:
                g_logs, g_text, g_time = agent.analyze_pipeline(temp_path, lambda p, t: prog_g.progress(p, text=t), engine="Gemini")
                prog_g.progress(85, text="Structuring data (RAG)...")

                g_incident = _build_incident_payload(video_path=Path(temp_path), clip_id=uploaded_file.name, logs=g_logs, analyst_text=g_text)
                g_retrieval = retrieve_for_incident(g_incident, index_dir=INDEX_DIR, top_k=DEFAULT_TOP_K)
                g_violations = map_retrieval_payload(g_retrieval, rule_pack_dir=DEFAULT_RULE_PACK_DIR)
                g_teachable = build_coaching_payload(g_violations)
                g_report = build_report(g_incident, g_retrieval, g_violations, g_teachable)

                # Save outputs to eval_data_dir
                with open(os.path.join(eval_data_dir, f"{base_name}_gemini_pred.json"), "w", encoding="utf-8") as f:
                    json.dump(g_logs, f, indent=2)
                with open(os.path.join(eval_data_dir, f"{base_name}_gemini_pred_report.txt"), "w", encoding="utf-8") as f:
                    f.write(g_text)

                st.session_state['g_logs'] = g_logs
                st.session_state['g_text'] = g_text
                st.session_state['g_time'] = g_time
                st.session_state['g_report'] = g_report
                st.session_state['g_error'] = False
                prog_g.progress(100, text="Complete!")
                time.sleep(0.5)
                prog_g.empty()
            except Exception as e:
                st.session_state['g_error'] = True
                prog_g.empty()
                with exec_col1:
                    st.error(f"Execution Failed: {e}")

        def run_openai():
            try:
                o_logs, o_text, o_time = agent.analyze_pipeline(temp_path, lambda p, t: prog_o.progress(p, text=t), engine="OpenAI")
                prog_o.progress(85, text="Structuring data (RAG)...")

                o_incident = _build_incident_payload(video_path=Path(temp_path), clip_id=uploaded_file.name, logs=o_logs, analyst_text=o_text)
                o_retrieval = retrieve_for_incident(o_incident, index_dir=INDEX_DIR, top_k=DEFAULT_TOP_K)
                o_violations = map_retrieval_payload(o_retrieval, rule_pack_dir=DEFAULT_RULE_PACK_DIR)
                o_teachable = build_coaching_payload(o_violations)
                o_report = build_report(o_incident, o_retrieval, o_violations, o_teachable)

                # Save outputs to eval_data_dir
                with open(os.path.join(eval_data_dir, f"{base_name}_openai_pred.json"), "w", encoding="utf-8") as f:
                    json.dump(o_logs, f, indent=2)
                with open(os.path.join(eval_data_dir, f"{base_name}_openai_pred_report.txt"), "w", encoding="utf-8") as f:
                    f.write(o_text)

                st.session_state['o_logs'] = o_logs
                st.session_state['o_text'] = o_text
                st.session_state['o_time'] = o_time
                st.session_state['o_report'] = o_report
                st.session_state['o_error'] = False
                prog_o.progress(100, text="Complete!")
                time.sleep(0.5)
                prog_o.empty()
            except Exception as e:
                st.session_state['o_error'] = True
                prog_o.empty()
                with exec_col2:
                    st.error(f"Execution Failed: {e}")

        # --- START CONCURRENT EXECUTION ---
        t1 = threading.Thread(target=run_gemini)
        t2 = threading.Thread(target=run_openai)

        # Attach Streamlit context to the threads so st.progress works
        add_script_run_ctx(t1)
        add_script_run_ctx(t2)

        t1.start()
        t2.start()

        # Wait for both engines to finish before moving on
        t1.join()
        t2.join()

        # Optionally: Run evaluation script for this video, only if ground truth exists
        import subprocess
        base_truth_json = os.path.join(eval_data_dir, f"{base_name}_truths.json")
        base_truth_report = os.path.join(eval_data_dir, f"{base_name}_report.txt")
        if os.path.exists(base_truth_json) and os.path.exists(base_truth_report):
            try:
                subprocess.run([
                    "python", "evaluate.py", eval_data_dir
                ], cwd=os.path.dirname(os.path.abspath(__file__)), check=True)
            except Exception as e:
                st.warning(f"Evaluation script failed: {e}")
        else:
            st.info("No ground truth data found for this video. Evaluation metrics will be shown only when ground truth is available.")

        st.session_state['file_processed'] = True
        st.rerun()

# --- DISPLAY RESULTS COMPONENT ---
def render_engine_results(engine_name, prefix, base_name):
    """Helper function to render the exact same UI components for either engine"""
    if st.session_state.get(f'{prefix}_error'):
        st.warning(f"⚠️ The {engine_name} pipeline did not complete successfully due to an error upstream.")
        return
        
    report = st.session_state.get(f'{prefix}_report')
    raw_text = st.session_state.get(f'{prefix}_text')
    logs = st.session_state.get(f'{prefix}_logs')
    proc_time = st.session_state.get(f'{prefix}_time')
    
    if not report: 
        return

    summary = report.get('summary', {})
    
    # Dynamic Risk Header
    risk = summary.get('overall_risk', 'unknown').lower()
    risk_color = "🔴" if risk == "high" else "🟡" if risk == "medium" else "🟢"
    st.subheader(f"Risk Level: {risk_color} {risk.upper()}")
    
    # 1. Top Findings Section
    st.markdown("#### Top Safety Findings")
    findings = summary.get('top_findings', [])
    if not findings:
        st.success("✅ No safety violations detected based on standard operating procedures.")
    else:
        for finding in findings:
            # Added prefix to expander titles to avoid identical key clashes in Streamlit
            with st.expander(f"⚠️ [{engine_name}] {finding.get('label')} ({finding.get('severity', '').upper()})", expanded=True):
                st.write(f"**Violation Code:** `{finding.get('violation_code')}`")
                st.write(f"**Confidence:** {finding.get('confidence')}")
                
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
        st.markdown("#### Recommended Immediate Actions")
        for action in actions:
            st.markdown(f"- 🛠️ {action}")

    st.markdown("<br>", unsafe_allow_html=True)
    
    # 3. VLM Narrative Breakdown
    vlm_summary = extract_section(raw_text, "OVERALL INCIDENT SUMMARY")
    vlm_sequence = extract_section(raw_text, "COMBINED NARRATIVE SEQUENCE")
    vlm_root_cause = extract_section(raw_text, "COMPREHENSIVE ROOT CAUSE OBSERVATION")

    with st.expander(f"📝 View {engine_name} Incident Narrative"):
        if not vlm_summary and not vlm_sequence:
            st.write(raw_text)
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
        
    with st.expander(f"🔎 View {engine_name} Frame Logs"):
        st.json(logs)
    
    # 4. Export PDF 
    pdf_filename = f"safety_report_{prefix}.pdf"
    create_pdf_report(report, raw_text, pdf_filename)
    
    with open(pdf_filename, "rb") as pdf_file:
        st.download_button(
            label=f"Export {engine_name} PDF Report",
            data=pdf_file,
            file_name=f"{engine_name}_Preliminary_Report.pdf",
            mime="application/pdf",
            key=f"dl_{prefix}" # Unique key for each button
        )
        
    # --- Automated Evaluation Section ---
    st.markdown("---")
    st.markdown("#### 📊 Evaluation Metrics")
    st.metric(f"⏱️ {engine_name} Processing Time", f"{proc_time} seconds")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    eval_folder = os.path.join(current_dir, "eval_data", base_name)
    eval_json_path = os.path.join(eval_folder, f"{base_name}_eval.json")
    truth_json_path = os.path.join(eval_folder, f"{base_name}_truths.json")
    truth_report_path = os.path.join(eval_folder, f"{base_name}_report.txt")

    if os.path.exists(eval_json_path):
        try:
            with open(eval_json_path, "r", encoding="utf-8") as f:
                eval_data = json.load(f)
        except json.JSONDecodeError as e:
            st.warning(f"Evaluation file is corrupted and could not be read: {e}")
            eval_data = None
    if os.path.exists(eval_json_path) and eval_data is not None:
        st.markdown("---")
        st.markdown("#### 📄 Evaluation Metrics for This Video")
        eval_col1, eval_col2 = st.columns(2)
        with eval_col1:
            st.markdown("**Phase 1: Observer**")
            obs = eval_data.get("observer")
            if obs:
                st.metric("Chunk Accuracy", f"{obs['accuracy'] * 100:.1f}%")
                st.metric("F1 Score", f"{obs['f1_score'] * 100:.1f}%")
                st.metric("Precision", f"{obs['precision']:.3f}")
                st.metric("Recall", f"{obs['recall']:.3f}")
                st.metric("Specificity", f"{obs['specificity']:.3f}")
                st.metric("NPV", f"{obs['npv']:.3f}")
                st.metric("MCC", f"{obs['mcc']:.3f}")
                st.metric("Cohen's Kappa", f"{obs['kappa']:.3f}")
                st.write(f"Confusion Matrix: {obs.get('confusion_matrix', {})}")
                st.write(f"False Positives: {obs.get('false_positives', [])}")
                st.write(f"False Negatives: {obs.get('false_negatives', [])}")
            else:
                st.info("No observer evaluation data.")
        with eval_col2:
            st.markdown("**Phase 2: Analyst**")
            ana = eval_data.get("analyst")
            if ana:
                st.metric("BERTScore (F1)", f"{ana['bert_f1']:.3f}")
                st.metric("METEOR Score", f"{ana['meteor_score']:.3f}")
                st.metric("BLEU", f"{ana['bleu']:.3f}")
                st.metric("ROUGE-1 F", f"{ana['rouge_1_f']:.3f}")
                st.metric("ROUGE-L F", f"{ana['rouge_l_f']:.3f}")
            else:
                st.info("No analyst evaluation data.")
    else:
        st.info(f"No evaluation data found for `{base_name}`.")

# --- RENDER THE SPLIT VIEW ---
if st.session_state.get('file_processed'):
    st.markdown("---")
    res_col1, res_col2 = st.columns(2)
    base_name = os.path.splitext(uploaded_file.name)[0]
    
    with res_col1:
        st.header("🔵 Gemini 3.1 Results")
        render_engine_results("Gemini", "g", base_name)
        
    with res_col2:
        st.header("🟢 OpenAI GPT-4o Results")
        render_engine_results("OpenAI", "o", base_name)
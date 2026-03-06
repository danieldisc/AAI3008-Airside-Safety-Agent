import streamlit as st
import os
import time
from vlm_agent import SafetyAgent
from report_gen import create_pdf_report

st.set_page_config(
    page_title="Changi Airside Safety Audit",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

with st.sidebar:
    st.markdown("### Configuration")
    st.markdown("---")
    st.markdown("**Enforcement Protocols**")
    rule_exclude = st.checkbox("Exclusion Zones (3m)", value=True)
    st.markdown("---")
    st.caption("System Version: v3.0 (Map-Reduce VLM Pipeline)")

st.title("Automated Incident Investigation")
st.markdown("Upload CCTV footage (up to 5 mins). The system chunks the video and utilizes an Observer AI to monitor states.")

uploaded_file = st.file_uploader("Source Footage (MP4/MOV)", type=["mp4", "mov"])

if uploaded_file is not None:
    st.video(uploaded_file)
    
    if st.button("Run Compliance Analysis", type="primary"):
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        # Using st.progress to track the chunking process
        progress_bar = st.progress(0, text="Initializing Pipeline...")
        
        def update_progress(percent, text):
            progress_bar.progress(percent, text=text)
            
        try:
            agent = SafetyAgent()
            analysis_text = agent.analyze_pipeline(temp_path, progress_callback=update_progress)
            
            update_progress(100, text="Pipeline Execution Complete.")
            time.sleep(1)
            progress_bar.empty()
            
            st.session_state['analysis_result'] = analysis_text
            st.session_state['file_processed'] = True

        except Exception as e:
            st.error(f"Processing Error: {e}")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

if st.session_state.get('file_processed') and st.session_state.get('analysis_result'):
    st.markdown("---")
    st.subheader("Incident Report")
    
    st.markdown(f"""
    <div class="report-container">
        {st.session_state['analysis_result']}
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    pdf_filename = "safety_report.pdf"
    create_pdf_report(st.session_state['analysis_result'], pdf_filename)
    
    with open(pdf_filename, "rb") as pdf_file:
        st.download_button(
            label="Export PDF Report",
            data=pdf_file,
            file_name="Preliminary_Incident_Report.pdf",
            mime="application/pdf"
        )
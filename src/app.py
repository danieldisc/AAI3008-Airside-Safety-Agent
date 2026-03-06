import streamlit as st
import os
import time
from vlm_agent import SafetyAgent
from report_gen import create_pdf_report

# 1. Page Configuration
st.set_page_config(
    page_title="Changi Airside Safety Audit",
    page_icon="🛡️", # Kept one subtle icon for the browser tab only
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Corporate CSS (Strict & Clean)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: #e0e0e0;
    }

    /* Backgrounds */
    .stApp {
        background-color: #0f1116; 
    }
    
    /* Report Container */
    .report-container {
        background-color: #161b22;
        padding: 24px;
        border-radius: 6px;
        border: 1px solid #30363d;
        margin-top: 20px;
    }

    /* Headers */
    h1, h2, h3 {
        color: #ffffff;
        font-weight: 600;
        letter-spacing: -0.5px;
    }
    
    /* Buttons */
    .stButton>button {
        border-radius: 6px;
        font-weight: 500;
        height: 45px;
        border: 1px solid #30363d;
    }
    
    .stButton>button:hover {
        border-color: #8b949e;
        color: #ffffff;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0d1117;
        border-right: 1px solid #30363d;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# 3. Sidebar (Configuration Only)
with st.sidebar:
    st.markdown("### Configuration")
    st.markdown("---")
    
    st.markdown("**Enforcement Protocols**")
    rule_exclude = st.checkbox("Exclusion Zones (7.5m)", value=True)
    rule_ingest = st.checkbox("Ingestion Hazard Zones", value=True)
    rule_ppe = st.checkbox("PPE Compliance", value=True)
    
    st.markdown("---")
    st.caption("System Version: v2.0.4 (Stable)")
    st.caption("Authorized Use Only")

# 4. Main Application Interface
st.title("Automated Incident Investigation")
st.markdown("Upload CCTV footage to generate a preliminary safety audit report.")

# File Uploader
uploaded_file = st.file_uploader("Source Footage (MP4/MOV)", type=["mp4", "mov"])

# Main Logic Container
if uploaded_file is not None:
    # Video Preview
    st.video(uploaded_file)
    
    # Analysis Button
    if st.button("Run Compliance Analysis", type="primary"):
        
        # 1. Setup
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        progress_bar = st.progress(0, text="Initializing...")
        
        try:
            # 2. Upload
            progress_bar.progress(30, text="Uploading to Inference Engine...")
            
            # 3. Analyze
            agent = SafetyAgent()
            progress_bar.progress(50, text="Processing Visual Data...")
            
            # Call the AI
            analysis_text = agent.analyze_video(temp_path)
            
            # 4. Finalize
            progress_bar.progress(100, text="Complete")
            time.sleep(0.5)
            progress_bar.empty()
            
            # Save to Session State (Crucial for persistence)
            st.session_state['analysis_result'] = analysis_text
            st.session_state['file_processed'] = True

        except Exception as e:
            st.error(f"Processing Error: {e}")
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)

# 5. Output Display Section (Persistent)
if st.session_state.get('file_processed') and st.session_state.get('analysis_result'):
    
    st.markdown("---")
    st.subheader("Incident Report")
    
    # The Report Text
    st.markdown(f"""
    <div class="report-container">
        {st.session_state['analysis_result']}
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # PDF Download
    pdf_filename = "safety_report.pdf"
    create_pdf_report(st.session_state['analysis_result'], pdf_filename)
    
    with open(pdf_filename, "rb") as pdf_file:
        st.download_button(
            label="Export PDF Report",
            data=pdf_file,
            file_name="Preliminary_Incident_Report.pdf",
            mime="application/pdf"
        )
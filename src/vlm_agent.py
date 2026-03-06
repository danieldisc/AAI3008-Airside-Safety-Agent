import os
import time
import google.generativeai as genai
from dotenv import load_dotenv

class SafetyAgent:
    def __init__(self):
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("API Key not found!")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-flash-latest")

    def analyze_video(self, video_path):
        """Uploads video and returns the safety analysis text."""
        
        # 1. Upload
        print(f"📤 Uploading {video_path}...")
        video_file = genai.upload_file(path=video_path)
        
        # 2. Wait for processing
        while video_file.state.name == "PROCESSING":
            time.sleep(1)
            video_file = genai.get_file(video_file.name)
            
        if video_file.state.name == "FAILED":
            raise ValueError("Video processing failed.")
            
        # 3. Analyze
        safety_rules = """
        CHANGI AIRSIDE SAFETY DIRECTIVE (CASD) - REF: CAAS/IATA-IGOM:
        1. EXCLUSION ZONES: Personnel must maintain a minimum Safety Buffer of 7.5 METERS from any active aircraft engine or propeller.
        2. INGESTION HAZARD: Entering the suction zone (front 5m radius) of a running engine is a Critical Safety Violation.
        3. JET BLAST: Personnel must remain 45m clear of the rear of an aircraft with running engines.
        4. PPE: High-Visibility Vests and Hearing Protection are mandatory in all Airside Operational Areas (AOA).
        """
        
        prompt = """
        You are a Senior Airside Safety Auditor. Analyze this CCTV footage carefully.
        
        TASK:
        Determine if the ground staff are complying with safety rules or violating them.
        
        CRITICAL INSTRUCTIONS FOR VISION ANALYSIS:
        1. DEPTH PERCEPTION: Be very careful with camera perspective. A person standing 10 meters *behind* an engine might look close in a 2D video. Look for shadows, feet position, and overlapping objects to judge true distance.
        2. CONTEXT: If a person is holding a marshalling wand, standing near a tug, or clearly walking on a designated path, they are likely COMPLIANT.
        3. VERDICT: 
           - If the person is safe, strictly state: "NO VIOLATION DETECTED."
           - Only flag a "VIOLATION" if there is clear, unambiguous evidence of danger (e.g., touching the plane, standing inside the engine intake zone).
        
        OUTPUT FORMAT:
        - Status: [SAFE / VIOLATION]
        - Reasoning: [Explain the depth/distance observations]
        - Rule Reference: [Cite rule if violated, otherwise N/A]
        """
        
        response = self.model.generate_content([video_file, safety_rules, prompt])
        
        # 4. Cleanup (Optional: deletes from cloud to save space)
        # genai.delete_file(video_file.name)
        
        return response.text
import os
import cv2
import json
import time
from PIL import Image
import google.generativeai as genai
from dotenv import load_dotenv

class SafetyAgent:
    def __init__(self):
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("API Key not found!")
        genai.configure(api_key=api_key)
        
        # Using 1.5 Flash as it is highly efficient for multimodal tasks
        self.model = genai.GenerativeModel("gemini-2.5-flash-lite")

    def _extract_frames(self, video_path, extract_fps=1):
        """Extracts frames from the video at a specified FPS."""
        cap = cv2.VideoCapture(video_path)
        original_fps = round(cap.get(cv2.CAP_PROP_FPS))
        frame_interval = max(1, original_fps // extract_fps)
        
        frames = []
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_interval == 0:
                # Convert BGR (OpenCV) to RGB (Pillow)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_frame)
                # Resize to save token/bandwidth costs while maintaining semantics
                pil_img.thumbnail((768, 768)) 
                frames.append(pil_img)
                
            frame_count += 1
            
        cap.release()
        return frames

    def analyze_pipeline(self, video_path, progress_callback=None):
        """
        The Map-Reduce Pipeline:
        1. Extract frames at 1 FPS.
        2. Chunk into 10-second (10 frame) blocks.
        3. Observer Agent logs the data (JSON).
        4. If violation found -> Analyst Agent writes report.
        """
        if progress_callback:
            progress_callback(10, "Extracting frames at 1 FPS...")
            
        frames = self._extract_frames(video_path, extract_fps=1)
        
        # Chunk frames into groups of 10 (10 seconds)
        chunk_size = 10
        chunks = [frames[i:i + chunk_size] for i in range(0, len(frames), chunk_size)]
        
        violation_found = False
        incident_chunk = None
        incident_log = None
        incident_time_start = 0
        
        # --- MAP PHASE: The Observer ---
        observer_prompt = """
        You are a Senior Airside Safety Auditor tracking system. You are looking at sequential frames representing 1-second intervals.
        
        CHANGI AIRSIDE SAFETY DIRECTIVE (CASD) - REF: CAAS/IATA-IGOM:
        1. EXCLUSION ZONES: Personnel must maintain a minimum Safety Buffer of 7.5 METERS from any active aircraft engine or propeller.
        2. INGESTION HAZARD: Entering the suction zone (front 5m radius) of a running engine is a Critical Safety Violation.
        3. JET BLAST: Personnel must remain 45m clear of the rear of an aircraft with running engines.
        4. PPE: High-Visibility Vests are mandatory in all Airside Operational Areas (AOA).
        
        CRITICAL INSTRUCTIONS FOR VISION ANALYSIS:
        1. DEPTH PERCEPTION: Be very careful with camera perspective. A person standing 10 meters *behind* an engine might look close in a 2D video. Look for shadows, feet position, and overlapping objects to judge true distance.
        2. CONTEXT: If a person is holding a marshalling wand, standing near a tug, or clearly walking on a designated path, take this into account for their actions, but distance rules to active engines STILL APPLY.
        
        TASK:
        Analyze these frames and output a strictly formatted JSON array. The array must contain exactly one object for each frame provided.
        
        Schema for each object:
        {
            "frame_index": [integer starting from 0],
            "propeller_active": [boolean - true if engine/propeller is visibly spinning/running],
            "person_detected": [boolean - true if ground crew is visible],
            "danger_zone_violation": [boolean - true if person breaches the 7.5m exclusion, 5m ingestion, or 45m jet blast zones of an ACTIVE engine],
            "ppe_violation": [boolean - true if person is missing High-Visibility Vest]
        }
        Return ONLY valid JSON. No markdown, no conversational text.
        """
        
        prompt = """
        You are a Senior Airside Safety Auditor. Analyze this CCTV footage carefully.
        
        TASK:
        Determine if the ground staff are complying with safety rules or violating them.
        
        CRITICAL INSTRUCTIONS FOR VISION ANALYSIS:
        1. DEPTH PERCEPTION: Be very careful with camera perspective. A person standing 10 meters *behind* an engine might look close in a 2D video. Look for shadows, feet position, and overlapping objects to judge true distance.
        2. CONTEXT: If a person is holding a marshalling wand, standing near a tug, or clearly walking on a designated path, take this into account for their actions, but distance rules to active engines STILL APPLY.
        
        TASK:
        Analyze these frames and output a strictly formatted JSON array. The array must contain exactly one object for each frame provided.
        
        Schema for each object:
        {
            "frame_index": [integer starting from 0],
            "propeller_active": [boolean - true if engine/propeller is visibly spinning/running],
            "person_detected": [boolean - true if ground crew is visible],
            "danger_zone_violation": [boolean - true if person breaches the 7.5m exclusion, 5m ingestion, or 45m jet blast zones of an ACTIVE engine],
            "ppe_violation": [boolean - true if person is missing High-Visibility Vest]
        }
        Return ONLY valid JSON. No markdown, no conversational text.
        """

        for idx, chunk in enumerate(chunks):
            if progress_callback:
                progress_callback(20 + int((idx / len(chunks)) * 50), f"Observer analyzing chunk {idx+1}/{len(chunks)}...")
                
            # Send the prompt and the array of PIL Images directly
            response = self.model.generate_content(
                [observer_prompt] + chunk,
                generation_config={"response_mime_type": "application/json"}
            )
            
            try:
                log_data = json.loads(response.text)
                
                # Scan log for trigger condition
                for state in log_data:
                    # Trigger if they enter the danger zone while active, OR if they are missing PPE
                    if (state.get("propeller_active") and state.get("danger_zone_violation")) or state.get("ppe_violation"):
                        violation_found = True
                        incident_chunk = chunk
                        incident_log = log_data
                        incident_time_start = idx * chunk_size
                        break
                        
            except json.JSONDecodeError:
                print(f"Warning: Chunk {idx} returned invalid JSON.")
                continue
                
            if violation_found:
                break
                
            # Optional: Sleep to respect free-tier API rate limits
            time.sleep(2) 

        # --- REDUCE PHASE: The Analyst ---
        if violation_found:
            if progress_callback:
                progress_callback(80, "Violation detected! Analyst is writing the narrative report...")
                
            analyst_prompt = f"""
            You are a Senior Airside Safety Auditor. A safety violation occurred starting at approximately {incident_time_start} seconds into the footage.
            
            CHANGI AIRSIDE SAFETY DIRECTIVE (CASD) - REF: CAAS/IATA-IGOM:
            1. EXCLUSION ZONES: 7.5m buffer from active engine/propeller.
            2. INGESTION HAZARD: 5m radius suction zone.
            3. JET BLAST: 45m clear of the rear.
            4. PPE: High-Visibility Vests mandatory.
            
            SYSTEM LOG FOR THIS 10-SECOND WINDOW:
            {json.dumps(incident_log, indent=2)}
            
            TASK:
            Review the attached frames (corresponding to the log above). Generate a formal narrative incident report explaining what happened.
            Only flag a violation based on clear, unambiguous evidence. 
            
            OUTPUT FORMAT (Use these exact headings):
            
            ### INCIDENT SUMMARY
            [Brief 2-sentence overview of the event]
            
            ### NARRATIVE SEQUENCE
            [Detailed chronological breakdown. Explain the depth/distance observations. Why did the crew member enter the danger zone? What actions were they performing?]
            
            ### RULE REFERENCE & VERDICT
            [Cite the specific CASD rule violated (e.g., Exclusion Zone, Ingestion Hazard) and state the final status: SAFE or VIOLATION]
            
            ### ROOT CAUSE OBSERVATION
            [Based on visual context, explain why it happened (e.g., retrieving dropped equipment, distracted, improper marshalling position)]
            """
            
            final_response = self.model.generate_content([analyst_prompt] + incident_chunk)
            return final_response.text
            
        else:
            if progress_callback:
                progress_callback(100, "Analysis complete.")
            return "### AUDIT COMPLETE\n\nNo safety violations (personnel within 3 meters of an active propeller) were detected in the provided footage."
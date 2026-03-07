import os
from pprint import pprint
import cv2
import json
import time
from PIL import Image
from google import genai
from google.genai import types
from dotenv import load_dotenv

class SafetyAgent:
    def __init__(self):
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("API Key not found!")
            
        # NEW SDK: Initialize the Client instead of global configuration
        self.client = genai.Client(api_key=api_key)
        
        # Define the model to use throughout the class
        self.model_name = "gemini-3.1-flash-lite-preview"

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
        
        # violation_found = False
        # incident_chunk = None
        # incident_log = None
        # incident_time_start = 0
        incidents = []  # List to hold all detected violations
        
        # --- MAP PHASE: The Observer ---
        observer_prompt = """
        You are a Senior Airside Safety Auditor tracking system. You are looking at sequential frames representing 1-second intervals.
        
        CHANGI AIRSIDE SAFETY DIRECTIVE (CASD) - REF: CAAS/IATA-IGOM:
        1. EXCLUSION ZONES: Personnel must maintain a minimum Safety Buffer of 7.5 METERS from any active aircraft engine or propeller.
        2. INGESTION HAZARD: Entering the suction zone (front 5m radius) of a running engine is a Critical Safety Violation.
        3. JET BLAST: Personnel must remain 45m clear of the rear of an aircraft with running engines.
        
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
            "danger_zone_violation": [boolean - true if person breaches the 7.5m exclusion, 5m ingestion, or 45m jet blast zones of an ACTIVE engine]
        }
        Return ONLY valid JSON. No markdown, no conversational text.
        """  

        for idx, chunk in enumerate(chunks):
            if progress_callback:
                progress_callback(20 + int((idx / len(chunks)) * 50), f"Observer analyzing chunk {idx+1}/{len(chunks)}...")
                
            # NEW SDK: Using client.models.generate_content and types.GenerateContentConfig
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[observer_prompt] + chunk,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                )
            )
            
            try:
                log_data = json.loads(response.text)
                
                chunk_has_violation = False
                for state in log_data:
                    # Trigger only if they enter the danger zone while active
                    if state.get("propeller_active") and state.get("danger_zone_violation"):
                        chunk_has_violation = True
                        break
                
                if chunk_has_violation:
                    incidents.append({
                        "time_start": idx * chunk_size,
                        "log": log_data,
                        "chunk": chunk
                    })
                        
            except json.JSONDecodeError:
                print(f"Warning: Chunk {idx} returned invalid JSON.")
                continue
            
            # Note: The 'if violation_found: break' statement has been removed
            # so the loop continues processing the entire video.
                
            # Sleep to respect free-tier API rate limits
            time.sleep(2) 

        # --- REDUCE PHASE: The Analyst ---
        if incidents:
            if progress_callback:
                progress_callback(80, f"Found {len(incidents)} violation windows! Synthesizing the complete storyline...")
                
            # Combine all flagged frames and logs into a single timeline
            combined_frames = []
            combined_logs = []
            
            for incident in incidents:
                combined_frames.extend(incident['chunk'])
                combined_logs.append({
                    "time_window_start_seconds": incident['time_start'],
                    "log": incident['log']
                })
                
            analyst_prompt = f"""
            You are a Senior Airside Safety Auditor. Safety violations occurred at multiple points during this footage.
            
            CHANGI AIRSIDE SAFETY DIRECTIVE (CASD) - REF: CAAS/IATA-IGOM:
            1. EXCLUSION ZONES: 7.5m buffer from active engine/propeller.
            2. INGESTION HAZARD: 5m radius suction zone.
            3. JET BLAST: 45m clear of the rear.
            
            COMBINED SYSTEM LOG FOR ALL FLAGGED INCIDENTS:
            {json.dumps(combined_logs, indent=2)}
            
            TASK:
            Review the attached frames (which contain all flagged moments) corresponding to the combined logs above. 
            Analyze all these parts as a whole and synthesize them into ONE single, cohesive narrative incident report.
            Tell the complete storyline of what the crew member(s) were doing across all these flagged moments.
            
            OUTPUT FORMAT (Use these exact headings):
            
            ### OVERALL INCIDENT SUMMARY
            [Brief 2-3 sentence overview of the entire sequence of events]
            
            ### COMBINED NARRATIVE SEQUENCE
            [Detailed chronological breakdown combining all flagged parts. Explain the depth/distance observations. Tell the full story of what led to these violations as a single timeline.]
            
            ### RULE REFERENCE & VERDICT
            [Cite the specific CASD rules violated across the footage and state the final status: VIOLATION]
            
            ### COMPREHENSIVE ROOT CAUSE OBSERVATION
            [Based on the complete visual context, explain the underlying reason for these repeated/extended violations (e.g., systemic disregard for exclusion zones, retrieving dropped equipment, distracted, improper marshalling position)]
            """
            
            final_response = self.client.models.generate_content(
                model=self.model_name,
                contents=[analyst_prompt] + combined_frames
            )
            
            if progress_callback:
                progress_callback(100, "Analysis complete.")
            return final_response.text
            
        else:
            if progress_callback:
                progress_callback(100, "Analysis complete.")
            return "### AUDIT COMPLETE\n\nNo safety violations were detected in the provided footage."
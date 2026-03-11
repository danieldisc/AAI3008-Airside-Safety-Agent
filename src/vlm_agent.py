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
        
    def _load_prompt(self, filename):
        """Helper method to load text from a file using absolute paths."""
        # 1. Get the directory where vlm_agent.py currently lives (the 'src' folder)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 2. Join it with 'prompts' and the specific filename
        filepath = os.path.join(current_dir, "prompts", filename)
        
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                return file.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"Prompt file not found at {filepath}. Please ensure it exists.")

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
        full_video_logs = []
        
        # --- MAP PHASE: The Observer ---
        observer_prompt = self._load_prompt("observer_prompt.txt")

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
                
                full_video_logs.extend(log_data) 
                
                chunk_has_violation = False
                for state in log_data:
                    if state.get("propeller_active") and state.get("danger_zone_violation"):
                        chunk_has_violation = True
                        break
                
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
                
            analyst_prompt = self._load_prompt("analyst_prompt.txt")
            
            final_response = self.client.models.generate_content(
                model=self.model_name,
                contents=[analyst_prompt] + combined_frames
            )
            
            if progress_callback:
                progress_callback(100, "Analysis complete.")
            return full_video_logs, final_response.text
            
        else:
            if progress_callback:
                progress_callback(100, "Analysis complete.")
            return "### AUDIT COMPLETE\n\nNo safety violations were detected in the provided footage."
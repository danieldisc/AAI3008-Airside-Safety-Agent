import os
from pprint import pprint
import cv2
import json
import time
import base64
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv

# Google Gemini
from google import genai
from google.genai import types

# OpenAI (DUAL-ENGINE UPGRADE)
from openai import OpenAI

class SafetyAgent:
    def __init__(self):
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("API Key not found!")
            
        # NEW SDK: Initialize the Client instead of global configuration
        self.client = genai.Client(api_key=api_key)
        
        # Define the model to use throughout the class
        # DANIEL'S FIX: Upgraded from the "Lite" preview to the full-powered model for better spatial reasoning!
        self.model_name = "gemini-2.5-flash" 
        
        # DUAL-ENGINE UPGRADE: Initialise OpenAI Client if the key exists
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key:
            self.openai_client = OpenAI(api_key=openai_api_key)
        else:
            self.openai_client = None
            
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

    # DUAL-ENGINE UPGRADE: Helper to translate images for GPT-4o
    def _pil_to_base64(self, img):
        """Translates a PIL Image into a Base64 text string for OpenAI."""
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

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
                # DANIEL'S FIX: Increased resolution to 1024x1024 so the AI isn't blind! 🦅
                pil_img.thumbnail((1024, 1024)) 
                frames.append(pil_img)
                
            frame_count += 1
            
        cap.release()
        return frames

    # DUAL-ENGINE UPGRADE: Added 'engine' parameter to easily switch between AI models
    def analyze_pipeline(self, video_path, progress_callback=None, engine="Gemini"):
        """
        The Map-Reduce Pipeline:
        1. Extract frames at 1 FPS.
        2. Chunk into 10-second (10 frame) blocks.
        3. Observer Agent logs the data (JSON).
        4. If violation found -> Analyst Agent writes report.
        """
        if engine == "OpenAI" and not self.openai_client:
            return [], "Error: OpenAI API key not found in environment variables."

        if progress_callback:
            progress_callback(10, f"Extracting frames at 1 FPS for {engine}...")
            
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
        # DANIEL'S FIX: Ensure the AI knows exactly what JSON structure to output!
        observer_prompt += '\n\nOutput a JSON object with a "frames" key containing an array of objects for each frame.'

        for idx, chunk in enumerate(chunks):
            if progress_callback:
                progress_callback(20 + int((idx / len(chunks)) * 50), f"{engine} Observer analyzing chunk {idx+1}/{len(chunks)}...")
                
            try:
                # DUAL-ENGINE UPGRADE: Route the request to the correct AI Model
                if engine == "Gemini":
                    # NEW SDK: Using client.models.generate_content and types.GenerateContentConfig
                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=[observer_prompt] + chunk,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json"
                        )
                    )
                    response_text = response.text
                
                elif engine == "OpenAI":
                    content_list = [{"type": "text", "text": observer_prompt}]
                    for img in chunk:
                        base64_image = self._pil_to_base64(img)
                        content_list.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        })
                        
                    response = self.openai_client.chat.completions.create(
                        model="gpt-4o",
                        messages=[{"role": "user", "content": content_list}],
                        response_format={"type": "json_object"}
                    )
                    response_text = response.choices[0].message.content

                # Parse the JSON safely
                log_data_raw = json.loads(response_text)
                
                # DANIEL'S FIX: Normalise the JSON structure in case the AI wraps it differently
                log_data = log_data_raw.get("frames", log_data_raw) if isinstance(log_data_raw, dict) else log_data_raw
                
                full_video_logs.extend(log_data) 
                
                chunk_has_violation = False
                
                # DANIEL'S FIX: Removed Royston's duplicate loop and the dangerous "AND" logic gate. 
                # If they are in the danger zone, we flag it immediately! Paranoia mode engaged! 🚨
                for state in log_data:
                    if state.get("danger_zone_violation"):
                        chunk_has_violation = True
                        break
                
                if chunk_has_violation:
                    incidents.append({
                        "time_start": idx * chunk_size,
                        "log": log_data,
                        "chunk": chunk
                    })
                        
            except Exception as e:
                # DANIEL'S FIX: Enhanced error logging so it doesn't just crash silently.
                print(f"Warning: Chunk {idx} failed or returned invalid JSON using {engine}. Error: {e}")
                continue
            
            # Note: The 'if violation_found: break' statement has been removed
            # so the loop continues processing the entire video.
                
            # Sleep to respect free-tier API rate limits
            time.sleep(2) 

        # --- REDUCE PHASE: The Analyst ---
        if incidents:
            if progress_callback:
                progress_callback(80, f"Found {len(incidents)} violation windows! Synthesizing the complete storyline with {engine}...")
                
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
            
            # DANIEL'S FIX: Actually inject the combined JSON logs into the prompt so the Analyst isn't flying blind! 🦅
            analyst_prompt += f"\n\nCOMBINED SYSTEM LOG FOR ALL FLAGGED INCIDENTS:\n{json.dumps(combined_logs, indent=2)}"
            
            # DUAL-ENGINE UPGRADE: Route the narrative generation to the correct AI Model
            if engine == "Gemini":
                final_response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=[analyst_prompt] + combined_frames
                )
                final_text = final_response.text
                
            elif engine == "OpenAI":
                content_list = [{"type": "text", "text": analyst_prompt}]
                for img in combined_frames:
                    base64_image = self._pil_to_base64(img)
                    content_list.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                    })
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": content_list}]
                )
                final_text = response.choices[0].message.content
            
            if progress_callback:
                progress_callback(100, f"Analysis complete via {engine}.")
            return full_video_logs, final_text
            
        else:
            if progress_callback:
                progress_callback(100, f"Analysis complete via {engine}.")
            # DANIEL'S FIX: Always return a tuple with full_video_logs to prevent the app from crashing on safe videos! 🩹
            return full_video_logs, "### AUDIT COMPLETE\n\nNo safety violations were detected in the provided footage."
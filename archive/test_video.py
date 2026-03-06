import os
import time
from dotenv import load_dotenv
import google.generativeai as genai

# 1. Setup
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

# 2. Define the path to your video
# Make sure the filename matches EXACTLY what is in your data folder!
video_path = "data/video1.mp4"  # <--- CHANGE THIS if your file is named differently

print(f"🚀 Starting Video Analysis for: {video_path}")

# 3. Upload the video to Google's Cloud
# (Gemini can't see files on your laptop directly, so we send it up first)
print("📤 Uploading video file...")
video_file = genai.upload_file(path=video_path)
print(f"✅ Upload Complete: {video_file.name}")

# 4. Wait for processing
# Video takes a few seconds to "digest" on the server. We have to wait.
print("⏳ Processing video...", end="")
while video_file.state.name == "PROCESSING":
    print(".", end="")
    time.sleep(2)
    video_file = genai.get_file(video_file.name)

if video_file.state.name == "FAILED":
    raise ValueError(f"❌ Video processing failed: {video_file.state.name}")

print("\n✅ Video is ready!")

# 5. The "Safety Manual" (The Context)
safety_rules = """
CHANGI AIRSIDE SAFETY DIRECTIVE - MACHINERY HAZARDS:
1. EXCLUSION ZONES: Personnel must maintain a distance of 30cm from any moving machinery.
2. INTENTIONAL RISK: Attempting to touch moving parts, even without contact, is a Critical Safety Violation.
3. PPE: Hands must be protected when working near maintenance zones.
"""

# 6. The Prompt (The Question)
prompt = "You are a Safety Officer. Watch this video. Did the person violate the safety rules? Describe the action and cite the specific rule breached."

# 7. Ask Gemini
print("🤖 Analyzing...")
model = genai.GenerativeModel("gemini-flash-latest")

# We send the video object AND the text prompt together
response = model.generate_content([video_file, safety_rules, prompt])

# 8. Output
print("\n" + "="*40)
print("📝 INCIDENT REPORT")
print("="*40)
print(response.text)
print("="*40)

# 9. Cleanup (Optional but polite)
# genai.delete_file(video_file.name)
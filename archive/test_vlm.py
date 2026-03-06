import os
from dotenv import load_dotenv
import google.generativeai as genai

# 1. Load the secret key from .env
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("❌ Error: API Key not found. Check your .env file!")
else:
    print(f"✅ API Key found: {api_key[:5]}********")
    
    # 2. Configure the Brain
    genai.configure(api_key=api_key)
    
    # 3. Simple Text Test
    try:
        print("📡 Connecting to Gemini...")
        model = genai.GenerativeModel("gemini-flash-latest")
        response = model.generate_content("Hello! Are you ready for the Airside Safety project?")
        
        print("\n🤖 AI Response:")
        print(response.text)
        print("\n🎉 SYSTEM STATUS: OPERATIONAL")
    except Exception as e:
        print(f"\n❌ Connection Failed: {e}")
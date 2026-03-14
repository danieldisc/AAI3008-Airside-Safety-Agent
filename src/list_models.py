from google import genai
import os
from dotenv import load_dotenv

# Load the API key from your .env file
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Initialize the client
client = genai.Client(api_key=api_key)

print("Fetching available models...\n")

# Loop through and print the models
for model in client.models.list():
    print(f"Model ID: {model.name}")
    print(f"Description: {model.description}")
    print("-" * 50)
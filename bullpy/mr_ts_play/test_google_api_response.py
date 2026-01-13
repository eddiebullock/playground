#!/usr/bin/env python3
"""
Test script to inspect Google Gemini API response structure.
Run this to see exactly what the API returns.
"""
import os
import json
import requests
from dotenv import load_dotenv
from PIL import Image
import base64
import io

# Load environment variables from the known location
from pathlib import Path

# Load from the specific location
env_path = Path("/Users/eb2007/playground/bullpy/mr_ts_play/experiments/cam_human_like/training/.env")
if env_path.exists():
    load_dotenv(env_path, override=True)
    print(f"✅ Loaded .env from: {env_path}")
else:
    # Fallback: try relative path
    project_root = Path(__file__).parent
    env_path = project_root / "experiments" / "cam_human_like" / "training" / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=True)
        print(f"✅ Loaded .env from: {env_path}")
    else:
        print("⚠️  .env file not found, trying current directory...")
        load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

if not api_key:
    print("❌ GOOGLE_API_KEY or GEMINI_API_KEY not found")
    exit(1)

# Create a simple test image (1x1 pixel)
img = Image.new('RGB', (1, 1), color='red')
buffer = io.BytesIO()
img.save(buffer, format='PNG')
img_bytes = buffer.getvalue()
img_b64 = base64.b64encode(img_bytes).decode('utf-8')

# Build request
model = "gemini-2.5-flash"
url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

contents = [{
    "role": "user",
    "parts": [
        {"text": "Say hello"},
        {
            "inline_data": {
                "mime_type": "image/png",
                "data": img_b64
            }
        }
    ]
}]

payload = {
    "contents": contents,
    "generationConfig": {
        "maxOutputTokens": 200,
        "temperature": 0.0
    },
    "safetySettings": [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
    ]
}

print("=" * 80)
print("REQUEST PAYLOAD:")
print("=" * 80)
print(json.dumps(payload, indent=2))
print("\n")

# Make request
print("=" * 80)
print("MAKING API REQUEST...")
print("=" * 80)
response = requests.post(url, json=payload, timeout=30)
print(f"Status code: {response.status_code}")

if response.status_code != 200:
    print(f"❌ Error: {response.text}")
    exit(1)

response_data = response.json()

print("\n")
print("=" * 80)
print("FULL API RESPONSE:")
print("=" * 80)
print(json.dumps(response_data, indent=2, default=str))

print("\n")
print("=" * 80)
print("RESPONSE STRUCTURE ANALYSIS:")
print("=" * 80)
print(f"Top-level keys: {list(response_data.keys())}")

if "candidates" in response_data:
    print(f"Number of candidates: {len(response_data['candidates'])}")
    if len(response_data["candidates"]) > 0:
        candidate = response_data["candidates"][0]
        print(f"Candidate keys: {list(candidate.keys())}")
        print(f"Finish reason: {candidate.get('finishReason', 'N/A')}")
        
        if "content" in candidate:
            content = candidate["content"]
            print(f"Content type: {type(content)}")
            if isinstance(content, dict):
                print(f"Content keys: {list(content.keys())}")
                if "parts" in content:
                    print(f"Number of parts: {len(content['parts'])}")
                    for i, part in enumerate(content["parts"]):
                        print(f"  Part {i} type: {type(part)}")
                        if isinstance(part, dict):
                            print(f"  Part {i} keys: {list(part.keys())}")
                            if "text" in part:
                                print(f"  Part {i} text: {part['text'][:100]}...")
        else:
            print("❌ No 'content' field in candidate")

if "promptFeedback" in response_data:
    print(f"Prompt feedback: {response_data['promptFeedback']}")

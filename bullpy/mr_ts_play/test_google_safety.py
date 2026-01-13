#!/usr/bin/env python3
"""
Test script to check if safety settings work with Google Gemini API.
"""

import os
from dotenv import load_dotenv
from pathlib import Path
from PIL import Image
import io

# Load .env file
env_path = Path("experiments/cam_human_like/training/.env")
if env_path.exists():
    load_dotenv(env_path)

try:
    import google.generativeai as genai
    
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ GOOGLE_API_KEY not found")
        exit(1)
    
    print("✅ API Key found")
    print("\nTesting safety settings...\n")
    
    genai.configure(api_key=api_key)
    
    # Create a simple test image (white square)
    test_image = Image.new('RGB', (224, 224), color='white')
    img_bytes = io.BytesIO()
    test_image.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    
    # Test 1: Without safety settings
    print("Test 1: Without safety settings")
    print("-" * 60)
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(
            ["What emotion is shown in this image? Options: happy, sad, neutral. Answer with just one word.", img_bytes.getvalue()],
            generation_config={"max_output_tokens": 10, "temperature": 0.0}
        )
        print(f"✅ Response: {response.text}")
        if response.candidates:
            print(f"   Finish reason: {response.candidates[0].finish_reason}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\nTest 2: With safety settings (enum format)")
    print("-" * 60)
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        safety_settings = [
            {
                "category": genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                "threshold": genai.types.HarmBlockThreshold.BLOCK_NONE
            },
            {
                "category": genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                "threshold": genai.types.HarmBlockThreshold.BLOCK_NONE
            },
            {
                "category": genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                "threshold": genai.types.HarmBlockThreshold.BLOCK_NONE
            },
            {
                "category": genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                "threshold": genai.types.HarmBlockThreshold.BLOCK_NONE
            },
        ]
        response = model.generate_content(
            ["What emotion is shown in this image? Options: happy, sad, neutral. Answer with just one word.", img_bytes.getvalue()],
            generation_config={"max_output_tokens": 10, "temperature": 0.0},
            safety_settings=safety_settings
        )
        print(f"✅ Response: {response.text}")
        if response.candidates:
            print(f"   Finish reason: {response.candidates[0].finish_reason}")
            if hasattr(response.candidates[0], 'safety_ratings'):
                print(f"   Safety ratings: {response.candidates[0].safety_ratings}")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nTest 3: Check available HarmCategory and HarmBlockThreshold values")
    print("-" * 60)
    try:
        print("HarmCategory values:")
        for attr in dir(genai.types.HarmCategory):
            if not attr.startswith('_'):
                print(f"  - {attr}")
        print("\nHarmBlockThreshold values:")
        for attr in dir(genai.types.HarmBlockThreshold):
            if not attr.startswith('_'):
                print(f"  - {attr}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\nTest 4: Try string format")
    print("-" * 60)
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
        response = model.generate_content(
            ["What emotion is shown in this image? Options: happy, sad, neutral. Answer with just one word.", img_bytes.getvalue()],
            generation_config={"max_output_tokens": 10, "temperature": 0.0},
            safety_settings=safety_settings
        )
        print(f"✅ Response: {response.text}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
except ImportError:
    print("❌ google.generativeai not installed")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

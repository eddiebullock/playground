#!/usr/bin/env python3
"""
Test script to check which Google Gemini models are available.
"""

import os
from dotenv import load_dotenv
from pathlib import Path

# Load .env file
env_path = Path("experiments/cam_human_like/training/.env")
if env_path.exists():
    load_dotenv(env_path)

try:
    import google.generativeai as genai
    
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ GOOGLE_API_KEY or GEMINI_API_KEY not found in environment")
        exit(1)
    
    print(f"✅ API Key found (starts with: {api_key[:10]}...)")
    print("\nListing available models...\n")
    
    genai.configure(api_key=api_key)
    
    # List all available models
    try:
        models = genai.list_models()
        print("Available models:")
        print("=" * 60)
        for model in models:
            model_name = model.name.replace('models/', '')
            supports_generation = 'generateContent' in model.supported_generation_methods if hasattr(model, 'supported_generation_methods') else 'Unknown'
            print(f"  - {model_name}")
            if hasattr(model, 'supported_generation_methods'):
                print(f"    Methods: {model.supported_generation_methods}")
            print()
        
        print("=" * 60)
        print("\nModels that support generateContent (for vision):")
        vision_models = []
        for model in models:
            if hasattr(model, 'supported_generation_methods'):
                if 'generateContent' in model.supported_generation_methods:
                    model_name = model.name.replace('models/', '')
                    vision_models.append(model_name)
                    print(f"  ✅ {model_name}")
        
        if vision_models:
            print(f"\n💡 Try using one of these models: {vision_models[0]}")
        else:
            print("\n⚠️  No models found that support generateContent")
            
    except Exception as e:
        print(f"❌ Error listing models: {e}")
        print("\nTrying alternative approach...")
        
        # Try common model names
        test_models = [
            "gemini-pro",
            "gemini-pro-vision", 
            "gemini-1.5-pro",
            "gemini-1.5-flash",
            "gemini-1.0-pro",
            "gemini-1.0-pro-vision"
        ]
        
        print("\nTesting model names directly...")
        for model_name in test_models:
            try:
                model = genai.GenerativeModel(model_name)
                # Try a simple test
                response = model.generate_content("Say 'test'")
                print(f"✅ {model_name} - WORKS!")
                break
            except Exception as e:
                error_msg = str(e)
                if "404" in error_msg or "not found" in error_msg.lower():
                    print(f"❌ {model_name} - Not found")
                else:
                    print(f"⚠️  {model_name} - Error: {error_msg[:100]}")
    
    print("\n" + "="*60)
    print("Note: The google.generativeai package is deprecated.")
    print("Consider upgrading to google.genai package.")
    print("="*60)
    
except ImportError:
    print("❌ Google Generative AI package not installed.")
    print("Run: pip install google-generativeai")
    print("\nOr try the new package: pip install google-genai")
except Exception as e:
    print(f"❌ Error: {e}")

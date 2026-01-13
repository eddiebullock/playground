#!/usr/bin/env python3
"""
Test script to check which Anthropic models are available with your API key.
"""

import os
from dotenv import load_dotenv
from pathlib import Path

# Load .env file
env_path = Path("experiments/cam_human_like/training/.env")
if env_path.exists():
    load_dotenv(env_path)

try:
    import anthropic
    
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ ANTHROPIC_API_KEY not found in environment")
        exit(1)
    
    print(f"✅ API Key found (starts with: {api_key[:10]}...)")
    print("\nTesting different model names...\n")
    
    client = anthropic.Anthropic(api_key=api_key)
    
    # List of common Claude model names to try
    models_to_test = [
        "claude-3-5-sonnet-20240620",
        "claude-3-5-sonnet-20241022",
        "claude-3-opus-20240229",
        "claude-3-5-haiku-20241022",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307",
        "claude-3-5-sonnet",  # Without date
        "claude-sonnet-4-20250514",  # Newer format
    ]
    
    print("Testing model availability with a simple API call...\n")
    
    for model_name in models_to_test:
        try:
            # Try a minimal API call
            response = client.messages.create(
                model=model_name,
                max_tokens=10,
                messages=[{
                    "role": "user",
                    "content": "Say 'test'"
                }]
            )
            print(f"✅ {model_name} - WORKS!")
            print(f"   Response: {response.content[0].text[:50]}")
            break
        except Exception as e:
            error_msg = str(e)
            if "404" in error_msg or "not_found" in error_msg.lower():
                print(f"❌ {model_name} - Not found (404)")
            elif "401" in error_msg or "authentication" in error_msg.lower():
                print(f"⚠️  {model_name} - Authentication error (check API key)")
                break
            elif "403" in error_msg or "permission" in error_msg.lower():
                print(f"⚠️  {model_name} - Permission denied (check access)")
            else:
                print(f"⚠️  {model_name} - Error: {error_msg[:100]}")
    
    print("\n" + "="*60)
    print("Next steps:")
    print("1. Check Anthropic Console: https://console.anthropic.com/")
    print("2. Verify your API key has access to Claude models")
    print("3. Check billing/credits are available")
    print("4. Look for 'Models' or 'API Access' section in console")
    
except ImportError:
    print("❌ Anthropic package not installed. Run: pip install anthropic")
except Exception as e:
    print(f"❌ Error: {e}")

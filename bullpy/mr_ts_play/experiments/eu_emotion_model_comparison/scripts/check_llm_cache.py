#!/usr/bin/env python3
"""
Check LLM cache and recover any completed API calls.

This helps recover results if evaluation was interrupted.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

def check_cache(model_name: str = "gpt-4o-mini"):
    """Check what's in the LLM cache for a given model."""
    cache_dir = Path("experiments/eu_emotion_model_comparison/data/llm_cache")
    if not cache_dir.exists():
        print(f"Cache directory not found: {cache_dir}")
        return
    
    model_name_safe = model_name.replace('-', '_')
    cache_files = list(cache_dir.glob(f"{model_name_safe}_*.json"))
    
    print(f"Found {len(cache_files)} cache files for {model_name}")
    
    if len(cache_files) == 0:
        print("No cache files found. Either:")
        print("1. API calls failed before caching")
        print("2. Cache directory is different")
        print("3. Model name doesn't match")
        return
    
    # Analyze cache files
    total_cost = 0.0
    total_tokens = 0
    successful = 0
    failed = 0
    
    for cache_file in cache_files:
        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)
            
            if 'scores' in data:
                successful += 1
                total_cost += data.get('cost_usd', 0)
                total_tokens += data.get('input_tokens', 0) + data.get('output_tokens', 0)
            else:
                failed += 1
        except Exception as e:
            print(f"Error reading {cache_file}: {e}")
            failed += 1
    
    print(f"\nCache Analysis:")
    print(f"  Successful calls: {successful}")
    print(f"  Failed calls: {failed}")
    print(f"  Total cost: ${total_cost:.4f}")
    print(f"  Total tokens: {total_tokens:,}")
    
    if successful > 0:
        print(f"\n✅ You have {successful} successful API calls cached!")
        print(f"   These can be recovered. Run:")
        print(f"   python experiments/eu_emotion_model_comparison/scripts/recover_llm_results.py --model {model_name}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='gpt-4o-mini', help='Model name to check')
    args = parser.parse_args()
    
    check_cache(args.model)

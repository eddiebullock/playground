#!/usr/bin/env python3
"""
Estimate cost for LLM evaluation on full dataset.
This helps you decide if it's worth running.
"""

import json
from pathlib import Path

def estimate_cost():
    # Load previous results to calculate cost per sample
    prev_results = Path('results/llm_only_eu_emotion_google/results.json')
    
    if prev_results.exists():
        with open(prev_results) as f:
            data = json.load(f)
            prev_samples = len(data.get('predictions', []))
            print(f"Previous test: {prev_samples} samples")
            print(f"Previous cost: ~$1.04 (from your earlier message)")
            print()
            
            # Calculate cost per sample
            cost_per_sample = 1.04 / prev_samples
            print(f"Cost per sample: ${cost_per_sample:.4f}")
    else:
        print("⚠️  No previous results found - using rough estimate")
        cost_per_sample = 0.02  # Rough estimate: $0.02 per sample
    
    # Full dataset size
    full_dataset_size = 546
    estimated_total = cost_per_sample * full_dataset_size
    
    print()
    print("="*60)
    print("COST ESTIMATE FOR FULL DATASET")
    print("="*60)
    print(f"Dataset size: {full_dataset_size} samples")
    print(f"Frames per video: 4 (from config)")
    print(f"Model: Google Gemini 2.5 Flash")
    print()
    print(f"Estimated cost: ${estimated_total:.2f}")
    print()
    print("⚠️  IMPORTANT NOTES:")
    print("   - This is an ESTIMATE based on previous run")
    print("   - Actual cost may vary by ±20-30%")
    print("   - Caching is enabled (reduces cost for repeated calls)")
    print("   - Google Gemini pricing: ~$0.075 per 1M input tokens")
    print("   - Each sample: ~4 frames + text prompt (~1000-2000 tokens)")
    print()
    print("💡 RECOMMENDATION:")
    if estimated_total <= 2.0:
        print("   ✅ Cost is reasonable (~$1-2). Safe to proceed.")
    elif estimated_total <= 5.0:
        print("   ⚠️  Cost is moderate (~$2-5). Consider testing on smaller subset first.")
    else:
        print("   ❌ Cost is high (>$5). Recommend testing on smaller subset first.")
    print()
    print("="*60)
    
    return estimated_total

if __name__ == "__main__":
    estimate_cost()

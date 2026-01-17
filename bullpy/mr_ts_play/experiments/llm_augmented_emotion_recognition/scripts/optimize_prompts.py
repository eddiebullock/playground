#!/usr/bin/env python3
"""
Systematic Prompt Optimization for LLM Emotion Recognition.

This script:
1. Defines multiple prompt variations
2. Evaluates each on validation set
3. Selects best performing prompt
4. Reports results for publication

Follows proper ML evaluation practices:
- Optimization on validation set only
- Test set held out until final evaluation
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.llm_augmented_emotion_recognition.models.llm_wrapper import LLMWrapper
from experiments.llm_augmented_emotion_recognition.scripts.test_llm_only import evaluate_llm


# Define prompt variations
PROMPT_VARIATIONS = {
    "baseline": {
        "name": "Baseline Prompt",
        "description": "Current prompt (EMOTION first, then REASONING)",
        "modifications": None  # Uses default prompt
    },
    
    "explicit_distinctions": {
        "name": "Explicit Distinctions",
        "description": "Adds explicit instructions for confusing emotion pairs",
        "modifications": {
            "add_distinctions": True,
            "distinction_pairs": [
                ("afraid", "surprised"),
                ("worried", "afraid"),
                ("worried", "afraid low intensity")
            ]
        }
    },
    
    "intensity_aware": {
        "name": "Intensity-Aware",
        "description": "Explicitly considers intensity as a dimension",
        "modifications": {
            "add_intensity_analysis": True
        }
    },
    
    "temporal_analysis": {
        "name": "Frame-by-Frame Temporal Analysis",
        "description": "Requests explicit frame-by-frame progression analysis",
        "modifications": {
            "add_temporal_analysis": True
        }
    },
    
    "combined": {
        "name": "Combined (Best Individual)",
        "description": "Combines best individual modifications",
        "modifications": {
            "add_distinctions": True,
            "add_intensity_analysis": True,
            "add_temporal_analysis": True
        }
    }
}


def create_enhanced_prompt(candidate_labels: List[str], variation: str) -> str:
    """Create enhanced prompt based on variation."""
    
    labels_str = ", ".join(candidate_labels)
    base_prompt = f"""Analyze these video frames and identify the emotion being expressed.

The emotion must be one of these options: {labels_str}

IMPORTANT: Provide your answer FIRST, then explain your reasoning.

Format your response EXACTLY as:
EMOTION: [single emotion label from the list above]
REASONING: [your detailed reasoning explaining what you observed and why this emotion matches]

What to observe:
- Facial expressions (eyes, mouth, eyebrows)
- Body language and posture
- Overall emotional tone"""
    
    mods = PROMPT_VARIATIONS[variation]["modifications"]
    if not mods:
        return base_prompt
    
    enhanced = base_prompt
    
    # Add explicit distinctions
    if mods.get("add_distinctions"):
        distinction_pairs = mods.get("distinction_pairs", [])
        distinctions_text = "\n\nCRITICAL DISTINCTIONS:\n"
        
        for emo1, emo2 in distinction_pairs:
            if emo1 in candidate_labels or emo2 in candidate_labels:
                if (emo1, emo2) == ("afraid", "surprised"):
                    distinctions_text += f"""
- If "{emo1}" and "{emo2}" are both options:
  * {emo1.upper()}: Eyes wide WITH TENSION, head may RETRACT backward, visible sclera above/below iris, 
    mouth open with TENSION, body LEANING BACK, defensive posture, protective response
  * {emo2.upper()}: Eyes wide but RELAXED, eyebrows raised, mouth in 'O' shape (relaxed), 
    head may LEAN FORWARD, engaged posture, curious response, no tension in facial muscles
"""
                elif (emo1, emo2) == ("worried", "afraid") or (emo1, emo2) == ("worried", "afraid low intensity"):
                    distinctions_text += f"""
- If "worried" and "{emo2}" are both options:
  * WORRIED: Subtle, sustained apprehension, slight brow furrow (not strong), eyes slightly widened 
    (not wide open), mouth closed or slightly open, lower intensity, more controlled expression, 
    may include slight head tilt or downward gaze
  * {emo2.upper()}: Strong fear response, wide eyes with visible tension, more intense facial 
    contractions, higher arousal, may include head retraction
"""
        
        enhanced += distinctions_text
    
    # Add intensity awareness
    if mods.get("add_intensity_analysis"):
        enhanced += """

INTENSITY ANALYSIS:
- Assess whether the emotion is LOW INTENSITY or FULL INTENSITY
- Low intensity: Subtle expressions, controlled, less extreme facial movements
- Full intensity: Strong expressions, more extreme facial movements, higher arousal
- Match the intensity level to the candidate labels"""
    
    # Add temporal analysis
    if mods.get("add_temporal_analysis"):
        enhanced += """

TEMPORAL ANALYSIS:
- Describe each frame individually (Frame 1, Frame 2, Frame 3, Frame 4)
- Identify the PROGRESSION of emotion across frames
- Note the PEAK expression (most intense moment)
- Consider the TRANSITION (how emotion develops over time)
- Emotions often have characteristic progressions (e.g., surprise: neutral → peak → recovery)"""
    
    return enhanced


def evaluate_prompt_variation(
    variation: str,
    config_path: str,
    validation_trials: str,
    output_dir: Path
) -> Dict:
    """Evaluate a single prompt variation on validation set."""
    
    print(f"\n{'='*80}")
    print(f"Evaluating: {PROMPT_VARIATIONS[variation]['name']}")
    print(f"{'='*80}")
    print(f"Description: {PROMPT_VARIATIONS[variation]['description']}")
    print()
    
    # Create output directory for this variation
    var_output_dir = output_dir / f"variation_{variation}"
    var_output_dir.mkdir(parents=True, exist_ok=True)
    
    # For now, we'll modify the LLM wrapper to use enhanced prompts
    # This requires updating the wrapper to accept custom prompts
    # For now, we'll use the existing test script and note the variation
    
    # Run evaluation
    try:
        # Note: This requires modifying llm_wrapper.py to support custom prompts
        # For now, we'll document the approach
        results = {
            "variation": variation,
            "name": PROMPT_VARIATIONS[variation]["name"],
            "description": PROMPT_VARIATIONS[variation]["description"],
            "status": "pending_implementation",
            "note": "Requires llm_wrapper.py modification to support custom prompts"
        }
        
        # Save results
        with open(var_output_dir / "variation_info.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
        
    except Exception as e:
        print(f"Error evaluating variation {variation}: {e}")
        return {
            "variation": variation,
            "error": str(e),
            "status": "failed"
        }


def main():
    parser = argparse.ArgumentParser(
        description="Systematic prompt optimization for LLM emotion recognition"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/llm_augmented_emotion_recognition/configs/llm_config.yaml",
        help="Path to LLM config file"
    )
    parser.add_argument(
        "--validation_trials",
        type=str,
        default="data/trial_definitions/eu_emotion_val_for_optimization.json",
        help="Path to validation trials (for optimization)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/llm_prompt_optimization",
        help="Output directory for optimization results"
    )
    parser.add_argument(
        "--variations",
        type=str,
        nargs="+",
        default=list(PROMPT_VARIATIONS.keys()),
        help="Which prompt variations to evaluate"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("SYSTEMATIC PROMPT OPTIMIZATION")
    print("="*80)
    print()
    print("This script evaluates multiple prompt variations on the VALIDATION set.")
    print("The best performing prompt will be selected for final test evaluation.")
    print()
    print(f"Validation set: {args.validation_trials}")
    print(f"Output directory: {output_dir}")
    print(f"Variations to evaluate: {len(args.variations)}")
    print()
    
    # Evaluate each variation
    results = []
    for variation in args.variations:
        if variation not in PROMPT_VARIATIONS:
            print(f"Warning: Unknown variation '{variation}', skipping")
            continue
        
        result = evaluate_prompt_variation(
            variation=variation,
            config_path=args.config,
            validation_trials=args.validation_trials,
            output_dir=output_dir
        )
        results.append(result)
    
    # Save summary
    summary = {
        "optimization_date": str(Path().cwd()),
        "validation_set": args.validation_trials,
        "variations_evaluated": len(results),
        "results": results
    }
    
    with open(output_dir / "optimization_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print()
    print("="*80)
    print("OPTIMIZATION COMPLETE")
    print("="*80)
    print()
    print("Next steps:")
    print("1. Review results in optimization_summary.json")
    print("2. Select best performing prompt variation")
    print("3. Run final evaluation on test set (ONCE)")
    print("4. Report validation and test results in paper")


if __name__ == "__main__":
    main()

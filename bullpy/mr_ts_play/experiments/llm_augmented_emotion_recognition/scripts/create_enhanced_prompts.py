#!/usr/bin/env python3
"""
Create enhanced prompts for emotion recognition.

This module provides functions to generate different prompt variations
for systematic optimization.
"""

from typing import List, Optional


def create_baseline_prompt(candidate_labels: List[str], include_reasoning: bool = True) -> str:
    """Create baseline prompt (current standard prompt)."""
    labels_str = ", ".join(candidate_labels)
    if include_reasoning:
        return f"""Analyze these video frames and identify the emotion being expressed.

The emotion must be one of these options: {labels_str}

IMPORTANT: Provide your answer FIRST, then explain your reasoning.

Format your response EXACTLY as:
EMOTION: [single emotion label from the list above]
REASONING: [your detailed reasoning explaining what you observed and why this emotion matches]

What to observe:
- Facial expressions (eyes, mouth, eyebrows)
- Body language and posture
- Overall emotional tone"""
    else:
        return f"""Analyze these video frames and identify the emotion being expressed.

The emotion must be one of these options: {labels_str}

Consider:
- Facial expressions (eyes, mouth, eyebrows)
- Body language and posture
- Overall emotional tone

Respond with ONLY the single emotion label from the list above."""


def create_explicit_distinctions_prompt(candidate_labels: List[str], include_reasoning: bool = True) -> str:
    """Create prompt with explicit distinctions for confusing emotion pairs."""
    labels_str = ", ".join(candidate_labels)
    
    # Check which distinctions are relevant
    distinctions_text = ""
    
    has_afraid = any("afraid" in label.lower() for label in candidate_labels)
    has_surprised = any("surprised" in label.lower() for label in candidate_labels)
    has_worried = any("worried" in label.lower() for label in candidate_labels)
    
    if has_afraid and has_surprised:
        distinctions_text += """
CRITICAL DISTINCTION - Afraid vs. Surprised:
- AFRAID: Eyes wide WITH TENSION (muscles contracted), visible sclera above AND below iris, 
  head may RETRACT backward, mouth open with TENSION (not relaxed), body LEANING BACK, 
  defensive posture, protective response, overall DEFENSIVE reaction
- SURPRISED: Eyes wide but RELAXED (no tension), eyebrows raised creating forehead wrinkles, 
  mouth in 'O' shape (relaxed, not tense), head may LEAN FORWARD (curiosity), 
  body FORWARD posture, engaged response, overall CURIOUS reaction
"""
    
    if has_worried and (has_afraid or any("afraid" in label.lower() for label in candidate_labels)):
        distinctions_text += """
CRITICAL DISTINCTION - Worried vs. Afraid:
- WORRIED: Subtle, sustained apprehension, slight brow furrow (not strong), 
  eyes slightly widened (not wide open), mouth closed or slightly open, 
  lower intensity, more controlled expression, may include slight head tilt or downward gaze, 
  overall SUBTLE concern
- AFRAID: Strong fear response, wide eyes with visible tension, more intense facial 
  contractions, higher arousal, may include head retraction, overall STRONG fear reaction
"""
    
    base_prompt = f"""Analyze these video frames and identify the emotion being expressed.

The emotion must be one of these options: {labels_str}

IMPORTANT: Provide your answer FIRST, then explain your reasoning.{distinctions_text}

Format your response EXACTLY as:
EMOTION: [single emotion label from the list above]
REASONING: [your detailed reasoning explaining what you observed and why this emotion matches]

What to observe:
- Facial expressions (eyes, mouth, eyebrows)
- Body language and posture
- Overall emotional tone"""
    
    return base_prompt


def create_intensity_aware_prompt(candidate_labels: List[str], include_reasoning: bool = True) -> str:
    """Create prompt that explicitly considers intensity as a dimension."""
    labels_str = ", ".join(candidate_labels)
    
    return f"""Analyze these video frames and identify the emotion being expressed.

The emotion must be one of these options: {labels_str}

IMPORTANT: Provide your answer FIRST, then explain your reasoning.

INTENSITY ANALYSIS:
- Assess whether the emotion is LOW INTENSITY or FULL INTENSITY
- Low intensity: Subtle expressions, controlled, less extreme facial movements, 
  restrained emotional display
- Full intensity: Strong expressions, more extreme facial movements, higher arousal, 
  more pronounced emotional display
- Match the intensity level to the candidate labels (e.g., "happy low intensity" vs "happy")

Format your response EXACTLY as:
EMOTION: [single emotion label from the list above]
REASONING: [your detailed reasoning explaining what you observed, the intensity level, and why this emotion matches]

What to observe:
- Facial expressions (eyes, mouth, eyebrows)
- Body language and posture
- Overall emotional tone
- Intensity level (low vs full)"""


def create_temporal_analysis_prompt(candidate_labels: List[str], include_reasoning: bool = True) -> str:
    """Create prompt that requests frame-by-frame temporal analysis."""
    labels_str = ", ".join(candidate_labels)
    
    return f"""Analyze these video frames and identify the emotion being expressed.

The emotion must be one of these options: {labels_str}

IMPORTANT: Provide your answer FIRST, then explain your reasoning.

TEMPORAL ANALYSIS APPROACH:
1. Describe each frame individually (Frame 1, Frame 2, Frame 3, Frame 4)
2. Identify the PROGRESSION of emotion across frames
3. Note the PEAK expression (most intense moment)
4. Consider the TRANSITION (how emotion develops over time)

Emotions often have characteristic progressions:
- Surprise: Neutral → Peak surprise → Recovery
- Fear: Neutral → Apprehension → Peak fear → Sustained
- Worried: Subtle concern that builds or sustains
- Happy: Gradual build-up or sustained positive expression

Format your response EXACTLY as:
EMOTION: [single emotion label from the list above]
REASONING: [frame-by-frame analysis with progression, peak identification, and transition description]

What to observe:
- Facial expressions (eyes, mouth, eyebrows) in each frame
- Body language and posture changes
- Overall emotional tone progression"""


def create_combined_prompt(candidate_labels: List[str], include_reasoning: bool = True) -> str:
    """Create combined prompt with all enhancements."""
    labels_str = ", ".join(candidate_labels)
    
    # Check which distinctions are relevant
    distinctions_text = ""
    
    has_afraid = any("afraid" in label.lower() for label in candidate_labels)
    has_surprised = any("surprised" in label.lower() for label in candidate_labels)
    has_worried = any("worried" in label.lower() for label in candidate_labels)
    
    if has_afraid and has_surprised:
        distinctions_text += """
CRITICAL DISTINCTION - Afraid vs. Surprised:
- AFRAID: Eyes wide WITH TENSION (muscles contracted), visible sclera above AND below iris, 
  head may RETRACT backward, mouth open with TENSION (not relaxed), body LEANING BACK, 
  defensive posture, protective response, overall DEFENSIVE reaction
- SURPRISED: Eyes wide but RELAXED (no tension), eyebrows raised creating forehead wrinkles, 
  mouth in 'O' shape (relaxed, not tense), head may LEAN FORWARD (curiosity), 
  body FORWARD posture, engaged response, overall CURIOUS reaction
"""
    
    if has_worried and (has_afraid or any("afraid" in label.lower() for label in candidate_labels)):
        distinctions_text += """
CRITICAL DISTINCTION - Worried vs. Afraid:
- WORRIED: Subtle, sustained apprehension, slight brow furrow (not strong), 
  eyes slightly widened (not wide open), mouth closed or slightly open, 
  lower intensity, more controlled expression, may include slight head tilt or downward gaze, 
  overall SUBTLE concern
- AFRAID: Strong fear response, wide eyes with visible tension, more intense facial 
  contractions, higher arousal, may include head retraction, overall STRONG fear reaction
"""
    
    return f"""Analyze these video frames and identify the emotion being expressed.

The emotion must be one of these options: {labels_str}

IMPORTANT: Provide your answer FIRST, then explain your reasoning.{distinctions_text}

INTENSITY ANALYSIS:
- Assess whether the emotion is LOW INTENSITY or FULL INTENSITY
- Low intensity: Subtle expressions, controlled, less extreme facial movements
- Full intensity: Strong expressions, more extreme facial movements, higher arousal
- Match the intensity level to the candidate labels

TEMPORAL ANALYSIS APPROACH:
1. Describe each frame individually (Frame 1, Frame 2, Frame 3, Frame 4)
2. Identify the PROGRESSION of emotion across frames
3. Note the PEAK expression (most intense moment)
4. Consider the TRANSITION (how emotion develops over time)

Format your response EXACTLY as:
EMOTION: [single emotion label from the list above]
REASONING: [frame-by-frame analysis with progression, intensity assessment, and why this emotion matches]

What to observe:
- Facial expressions (eyes, mouth, eyebrows) in each frame
- Body language and posture changes
- Overall emotional tone progression
- Intensity level (low vs full)"""


def get_prompt(variation: str, candidate_labels: List[str], include_reasoning: bool = True) -> str:
    """Get prompt for specified variation."""
    variations = {
        "baseline": create_baseline_prompt,
        "explicit_distinctions": create_explicit_distinctions_prompt,
        "intensity_aware": create_intensity_aware_prompt,
        "temporal_analysis": create_temporal_analysis_prompt,
        "combined": create_combined_prompt,
    }
    
    if variation not in variations:
        raise ValueError(f"Unknown variation: {variation}. Available: {list(variations.keys())}")
    
    return variations[variation](candidate_labels, include_reasoning)

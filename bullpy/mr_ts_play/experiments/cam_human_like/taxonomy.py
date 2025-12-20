"""
CAM Emotion Taxonomy

Defines the emotion taxonomy structure for the CAM Face-Voice Battery,
including emotion groups, CAM levels, and allowable foil groups.

Based on the CAM methodology (Golan et al., 2006):
- 24 emotion groups total
- Levels 4, 5, and 6 (adult levels)
- Foils should be from different emotion groups than target
- Foils should match valence or interpersonal theme for appropriate difficulty

This taxonomy maps emotions to their groups and defines which foil groups
are appropriate for each target emotion.
"""

from typing import Dict, List, Set, Optional, Tuple


# CAM Emotion Groups (24 groups total)
EMOTION_GROUPS = {
    "angry": ["angry", "annoyed", "furious", "resentful", "hostile", "irritated", "enraged"],
    "unfriendly": ["unfriendly", "hostile", "cold", "distant", "aloof", "antagonistic"],
    "sad": ["sad", "sorrowful", "melancholic", "depressed", "downhearted", "dejected"],
    "hurt": ["hurt", "wounded", "injured", "damaged", "bruised", "pained"],
    "happy": ["happy", "joyful", "cheerful", "glad", "pleased", "delighted", "elated"],
    "friendly": ["friendly", "warm", "welcoming", "affectionate", "kind", "caring"],
    "afraid": ["afraid", "frightened", "scared", "terrified", "anxious", "worried"],
    "disgusted": ["disgusted", "revolted", "repulsed", "appalled", "sickened"],
    "surprised": ["surprised", "shocked", "amazed", "astonished", "startled"],
    "proud": ["proud", "triumphant", "victorious", "accomplished", "successful"],
    "ashamed": ["ashamed", "embarrassed", "humiliated", "mortified", "disgraced"],
    "submissive": ["submissive", "subservient", "obedient", "compliant", "yielding"],
    "dominant": ["dominant", "authoritative", "commanding", "controlling", "powerful"],
    "interested": ["interested", "curious", "intrigued", "engaged", "attentive"],
    "bored": ["bored", "uninterested", "apathetic", "indifferent", "disengaged"],
    "contemptuous": ["contemptuous", "disdainful", "scornful", "derisive", "mocking"],
    "grateful": ["grateful", "thankful", "appreciative", "indebted", "obliged"],
    "jealous": ["jealous", "envious", "covetous", "resentful", "begrudging"],
    "guilty": ["guilty", "remorseful", "regretful", "penitent", "contrite"],
    "relieved": ["relieved", "comforted", "reassured", "eased", "calmed"],
    "disappointed": ["disappointed", "let-down", "disillusioned", "disheartened"],
    "hopeful": ["hopeful", "optimistic", "confident", "expectant", "positive"],
    "lonely": ["lonely", "isolated", "abandoned", "forsaken", "deserted"],
    "loving": ["loving", "affectionate", "adoring", "devoted", "fond"],
    # CAM 20 concepts additions
    "appalled": ["appalled", "disgusted", "revolted", "shocked", "horrified"],
    "appealing": ["appealing", "attractive", "enticing", "alluring", "charming"],
    "confronted": ["confronted", "challenged", "faced", "accosted", "cornered"],
    "distaste": ["distaste", "disgust", "revulsion", "aversion", "repugnance"],
    "empathic": ["empathic", "understanding", "compassionate", "sympathetic", "caring"],
    "exonerated": ["exonerated", "cleared", "absolved", "vindicated", "acquitted"],
    "grave": ["grave", "serious", "solemn", "somber", "earnest"],
    "guarded": ["guarded", "cautious", "wary", "defensive", "protective"],
    "insincere": ["insincere", "fake", "dishonest", "deceptive", "false"],
    "intimate": ["intimate", "close", "personal", "private", "familiar"],
    "lured": ["lured", "enticed", "tempted", "attracted", "seduced"],
    "mortified": ["mortified", "humiliated", "ashamed", "embarrassed", "disgraced"],
    "nostalgic": ["nostalgic", "sentimental", "wistful", "homesick", "reminiscent"],
    "reassured": ["reassured", "comforted", "consoled", "soothed", "calmed"],
    "resentful": ["resentful", "bitter", "indignant", "aggrieved", "hostile"],
    "stern": ["stern", "strict", "harsh", "severe", "rigid"],
    "subdued": ["subdued", "quiet", "restrained", "muted", "suppressed"],
    "subservient": ["subservient", "submissive", "obedient", "servile", "deferential"],
    "uneasy": ["uneasy", "anxious", "worried", "uncomfortable", "restless"],
    "vibrant": ["vibrant", "energetic", "lively", "dynamic", "animated"],
}


# CAM Levels (4, 5, 6 are adult levels)
# Level 4: Basic complex emotions
# Level 5: More nuanced emotions
# Level 6: Most subtle/complex emotions
CAM_LEVELS = {
    4: [
        "happy", "sad", "angry", "afraid", "surprised", "disgusted",
        "proud", "ashamed", "grateful", "jealous", "guilty", "relieved",
    ],
    5: [
        "resentful", "subservient", "contemptuous", "disappointed", "hopeful",
        "hurt", "unfriendly", "friendly", "interested", "bored", "lonely",
        "loving", "dominant", "submissive",
        # CAM 20 concepts (level 5)
        "distaste", "exonerated", "grave", "guarded", "insincere", "intimate",
        "lured", "mortified", "nostalgic", "reassured", "stern", "subdued",
    ],
    6: [
        "humiliating", "agonizing", "triumphant", "devastated", "exhilarated",
        # CAM 20 concepts (level 6)
        "uneasy",
    ],
}


# Taxonomy dictionary: maps each emotion to its group, level, and allowable foil groups
CAM_TAXONOMY: Dict[str, Dict[str, any]] = {}


def _build_taxonomy():
    """Build the complete taxonomy dictionary from emotion groups and levels."""
    global CAM_TAXONOMY
    
    # Build reverse mapping: emotion -> group
    emotion_to_group = {}
    for group, emotions in EMOTION_GROUPS.items():
        for emotion in emotions:
            emotion_to_group[emotion] = group
    
    # Build reverse mapping: emotion -> level
    emotion_to_level = {}
    for level, groups in CAM_LEVELS.items():
        for group in groups:
            if group in EMOTION_GROUPS:
                for emotion in EMOTION_GROUPS[group]:
                    emotion_to_level[emotion] = level
    
    # Define foil group relationships based on CAM methodology
    # Foils should be from different groups but match valence/theme
    foil_group_mappings = {
        "angry": ["unfriendly", "sad", "hurt", "contemptuous"],  # Negative valence, interpersonal
        "unfriendly": ["angry", "sad", "hurt", "contemptuous"],
        "sad": ["hurt", "unfriendly", "disappointed", "lonely"],
        "hurt": ["sad", "unfriendly", "disappointed", "ashamed"],
        "happy": ["friendly", "proud", "hopeful", "loving"],
        "friendly": ["happy", "loving", "grateful", "hopeful"],
        "afraid": ["anxious", "worried", "hurt", "disappointed"],
        "disgusted": ["contemptuous", "unfriendly", "ashamed", "hurt"],
        "surprised": ["interested", "afraid", "hopeful", "disappointed"],
        "proud": ["happy", "triumphant", "confident", "hopeful"],
        "ashamed": ["hurt", "sad", "disappointed", "guilty"],
        "submissive": ["subservient", "afraid", "ashamed", "hurt"],
        "dominant": ["proud", "confident", "angry", "contemptuous"],
        "interested": ["curious", "hopeful", "surprised", "engaged"],
        "bored": ["disappointed", "uninterested", "sad", "lonely"],
        "contemptuous": ["unfriendly", "angry", "disgusted", "dominant"],
        "grateful": ["happy", "friendly", "loving", "relieved"],
        "jealous": ["angry", "hurt", "resentful", "disappointed"],
        "guilty": ["ashamed", "hurt", "sad", "remorseful"],
        "relieved": ["happy", "grateful", "comforted", "hopeful"],
        "disappointed": ["sad", "hurt", "disappointed", "let-down"],
        "hopeful": ["happy", "optimistic", "confident", "interested"],
        "lonely": ["sad", "hurt", "disappointed", "abandoned"],
        "loving": ["friendly", "happy", "affectionate", "grateful"],
        # CAM 20 concepts foil mappings
        "appalled": ["disgusted", "ashamed", "hurt", "mortified"],
        "appealing": ["interested", "hopeful", "lured", "friendly"],
        "confronted": ["afraid", "uneasy", "guarded", "hurt"],
        "distaste": ["disgusted", "appalled", "unfriendly", "hurt"],
        "empathic": ["friendly", "loving", "caring", "reassured"],
        "exonerated": ["relieved", "reassured", "happy", "grateful"],
        "grave": ["sad", "serious", "subdued", "nostalgic"],
        "guarded": ["afraid", "uneasy", "confronted", "subdued"],
        "insincere": ["unfriendly", "stern", "contemptuous", "hurt"],
        "intimate": ["loving", "friendly", "close", "empathic"],
        "lured": ["interested", "appealing", "tempted", "hopeful"],
        "mortified": ["ashamed", "appalled", "hurt", "sad"],
        "nostalgic": ["sad", "grave", "subdued", "lonely"],
        "reassured": ["relieved", "exonerated", "comforted", "happy"],
        "resentful": ["angry", "unfriendly", "hurt", "stern"],
        "stern": ["unfriendly", "insincere", "grave", "subdued"],
        "subdued": ["sad", "grave", "nostalgic", "submissive"],
        "subservient": ["submissive", "subdued", "afraid", "guarded"],
        "uneasy": ["afraid", "confronted", "guarded", "worried"],
        "vibrant": ["happy", "hopeful", "energetic", "lively"],
    }
    
    # Build complete taxonomy
    for emotion, group in emotion_to_group.items():
        level = emotion_to_level.get(emotion, 5)  # Default to level 5
        
        # Get foil groups for this emotion's group
        foil_groups = foil_group_mappings.get(group, [])
        
        # If no specific mapping, use opposite valence or similar groups
        if not foil_groups:
            if group in ["happy", "friendly", "proud", "hopeful", "loving", "grateful"]:
                # Positive valence -> use other positive or neutral
                foil_groups = ["happy", "friendly", "proud", "hopeful"]
            elif group in ["sad", "hurt", "disappointed", "lonely", "ashamed", "guilty"]:
                # Negative valence -> use other negative
                foil_groups = ["sad", "hurt", "unfriendly", "disappointed"]
            else:
                # Neutral/mixed -> use diverse groups
                foil_groups = ["sad", "happy", "afraid", "interested"]
        
        CAM_TAXONOMY[emotion] = {
            "group": group,
            "level": level,
            "foil_groups": foil_groups,
        }


# Initialize taxonomy
_build_taxonomy()


def get_emotion_info(emotion: str) -> Optional[Dict]:
    """
    Get taxonomy information for an emotion.
    
    Args:
        emotion: Emotion label string
    
    Returns:
        Dictionary with keys: group, level, foil_groups
        Returns None if emotion not in taxonomy
    """
    return CAM_TAXONOMY.get(emotion.lower())


def get_foil_candidates(
    target_emotion: str,
    available_emotions: Set[str],
    num_foils: int = 3,
    foil_levels: Optional[List[int]] = None,
) -> List[str]:
    """
    Get candidate foil emotions following CAM methodology.
    
    Rules (from Golan et al., 2006):
    1. Foils must be from different emotion groups than target
    2. Foils should be from levels 4 and 5 (as per original CAM)
       OR same/adjacent levels if foil_levels not specified
    3. Foils should match valence/interpersonal theme
    
    Args:
        target_emotion: The target/correct emotion
        available_emotions: Set of all available emotions in dataset
        num_foils: Number of foils needed (default 3)
        foil_levels: Optional list of allowed CAM levels for foils
                    (default: same or adjacent levels to target)
    
    Returns:
        List of foil emotion labels
    """
    target_info = get_emotion_info(target_emotion)
    if not target_info:
        # If emotion not in taxonomy, fall back to random selection
        # excluding target and same-group emotions
        available = list(available_emotions - {target_emotion})
        return available[:num_foils]
    
    target_group = target_info["group"]
    target_level = target_info["level"]
    allowed_foil_groups = target_info["foil_groups"]
    
    # Determine allowed foil levels
    if foil_levels is not None:
        allowed_levels = set(foil_levels)
    else:
        # Default: same or adjacent levels (±1)
        allowed_levels = {target_level - 1, target_level, target_level + 1}
        allowed_levels = {l for l in allowed_levels if l >= 4 and l <= 6}  # Valid CAM levels
    
    # Filter available emotions:
    # 1. Exclude target emotion
    # 2. Exclude emotions from same group as target
    # 3. Prefer emotions from allowed foil groups
    # 4. Prefer emotions from allowed levels
    
    candidates = []
    preferred_candidates = []
    
    for emotion in available_emotions:
        if emotion.lower() == target_emotion.lower():
            continue
        
        emotion_info = get_emotion_info(emotion)
        if not emotion_info:
            # Unknown emotion - add to general candidates
            candidates.append(emotion)
            continue
        
        emotion_group = emotion_info["group"]
        emotion_level = emotion_info["level"]
        
        # Skip if same group as target
        if emotion_group == target_group:
            continue
        
        # Check if in allowed levels
        level_match = emotion_level in allowed_levels
        
        if emotion_group in allowed_foil_groups and level_match:
            preferred_candidates.append(emotion)
        elif level_match:
            candidates.append(emotion)
        else:
            # Still add as backup if we don't have enough
            candidates.append(emotion)
    
    # Prioritize preferred candidates
    foil_list = preferred_candidates[:num_foils]
    
    # Fill remaining slots from general candidates
    if len(foil_list) < num_foils:
        remaining = num_foils - len(foil_list)
        foil_list.extend(candidates[:remaining])
    
    # If still not enough, use any available (except target)
    if len(foil_list) < num_foils:
        remaining = num_foils - len(foil_list)
        all_available = list(available_emotions - {target_emotion})
        for emotion in all_available:
            if emotion not in foil_list:
                foil_list.append(emotion)
                remaining -= 1
                if remaining == 0:
                    break
    
    return foil_list[:num_foils]


def validate_trial_foils(
    target_emotion: str,
    foil_emotions: List[str],
) -> Tuple[bool, List[str]]:
    """
    Validate that foils follow CAM methodology rules.
    
    Args:
        target_emotion: The target/correct emotion
        foil_emotions: List of foil emotions
    
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    target_info = get_emotion_info(target_emotion)
    if not target_info:
        errors.append(f"Target emotion '{target_emotion}' not in taxonomy")
        return len(errors) == 0, errors
    
    target_group = target_info["group"]
    
    # Check each foil
    for foil in foil_emotions:
        foil_info = get_emotion_info(foil)
        if not foil_info:
            errors.append(f"Foil emotion '{foil}' not in taxonomy")
            continue
        
        foil_group = foil_info["group"]
        
        # Rule 1: Foil cannot be from same group as target
        if foil_group == target_group:
            errors.append(
                f"Foil '{foil}' (group: {foil_group}) is from same group as "
                f"target '{target_emotion}' (group: {target_group})"
            )
    
    # Rule 2: Should have exactly 3 foils
    if len(foil_emotions) != 3:
        errors.append(f"Expected 3 foils, got {len(foil_emotions)}")
    
    # Rule 3: No duplicate foils
    if len(foil_emotions) != len(set(foil_emotions)):
        errors.append("Duplicate foils found")
    
    # Rule 4: Target should not be in foils
    if target_emotion.lower() in [f.lower() for f in foil_emotions]:
        errors.append(f"Target emotion '{target_emotion}' appears in foils")
    
    return len(errors) == 0, errors


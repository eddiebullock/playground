"""
The 20 CAM concepts from Golan et al. (2006) Table IV.

These are the final 20 concepts after validation:
- Started with 25 concepts (6 from level 4, 15 from level 5, 4 from level 6)
- 5 concepts excluded (3 from level 6, 2 from level 5) after validation
- 8 concepts had 1 invalid item each removed
- Final: 20 concepts with 5 items each (100 total trials)

Concept selection criteria:
- Selected from all 24 emotion groups
- Mainly subtle emotions
- Important for everyday social functioning
- Larger groups (unfriendly, sad) represented by 2 concepts each
"""

from typing import Optional

# The 20 CAM concepts (from Table IV in Golan et al., 2006)
CAM_20_CONCEPTS = [
    "appalled",
    "appealing",  # (asking for)
    "confronted",
    "distaste",
    "empathic",
    "exonerated",
    "grave",
    "guarded",
    "insincere",
    "intimate",
    "lured",
    "mortified",
    "nostalgic",
    "reassured",
    "resentful",
    "stern",
    "subdued",
    "subservient",
    "uneasy",
    "vibrant",
]

# Concept distribution by level (from original 25, before exclusions)
# Note: The paper doesn't specify which 5 were excluded, but we know:
# - 3 from level 6 were excluded
# - 2 from level 5 were excluded
# Original distribution: 6 from level 4, 15 from level 5, 4 from level 6
# Final distribution: 6 from level 4, 13 from level 5, 1 from level 6

CAM_CONCEPT_LEVELS = {
    # Level 4 concepts (6 total)
    "appalled": 4,
    "appealing": 4,
    "confronted": 4,
    "empathic": 4,
    "reassured": 4,
    "vibrant": 4,
    
    # Level 5 concepts (13 total)
    "distaste": 5,
    "exonerated": 5,
    "grave": 5,
    "guarded": 5,
    "insincere": 5,
    "intimate": 5,
    "lured": 5,
    "mortified": 5,
    "nostalgic": 5,
    "resentful": 5,
    "stern": 5,
    "subdued": 5,
    "subservient": 5,
    
    # Level 6 concepts (1 total - 3 were excluded)
    "uneasy": 6,
}

# Emotion group mapping (based on CAM taxonomy)
CAM_CONCEPT_GROUPS = {
    "appalled": "disgusted",
    "appealing": "interested",  # asking for something
    "confronted": "afraid",
    "distaste": "disgusted",
    "empathic": "friendly",
    "exonerated": "relieved",
    "grave": "sad",
    "guarded": "afraid",
    "insincere": "unfriendly",
    "intimate": "loving",
    "lured": "interested",
    "mortified": "ashamed",
    "nostalgic": "sad",
    "reassured": "relieved",
    "resentful": "angry",
    "stern": "unfriendly",
    "subdued": "submissive",
    "subservient": "submissive",
    "uneasy": "afraid",
    "vibrant": "happy",
}


def get_cam_20_concepts():
    """Return the list of 20 CAM concepts."""
    return CAM_20_CONCEPTS.copy()


def is_cam_concept(emotion: str) -> bool:
    """Check if an emotion is one of the 20 CAM concepts."""
    return emotion.lower() in [c.lower() for c in CAM_20_CONCEPTS]


def get_concept_level(concept: str) -> Optional[int]:
    """Get the CAM level for a concept."""
    return CAM_CONCEPT_LEVELS.get(concept.lower())


def get_concept_group(concept: str) -> Optional[str]:
    """Get the emotion group for a concept."""
    return CAM_CONCEPT_GROUPS.get(concept.lower())


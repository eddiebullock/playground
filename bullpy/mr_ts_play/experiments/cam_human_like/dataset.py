"""
CAM dataset utilities for parsing filenames and working with the dataset.
"""

import re
from typing import Dict, Optional


def _parse_filename(filename: str) -> Optional[Dict]:
    """
    Parse CAM video filename to extract metadata.
    
    Format: {scenario_id}{actor}{number}{V|T}{emotion}.mov
    Example: 0100104M1Vhumiliating.mov
    
    Args:
        filename: Video filename (e.g., "0100104M1Vhumiliating.mov")
    
    Returns:
        Dictionary with parsed metadata:
        {
            "scenario_id": "0100104",
            "actor": "M",
            "instance_num": "1",
            "modality": "V",  # V = Visual (face), T = Textual (voice)
            "emotion": "humiliating",
        }
        Returns None if parsing fails.
    """
    base = filename.replace(".mov", "")
    
    # Extract emotion (last part after V/T, may contain hyphens)
    match = re.search(r'([VT])([a-z]+(?:-[a-z]+)*)$', base)
    if not match:
        return None
    
    modality = match.group(1)  # V or T
    emotion = match.group(2)
    prefix = base[:match.start()]
    
    # Extract scenario ID (first 7 digits)
    scenario_match = re.match(r'^(\d{7})', prefix)
    if not scenario_match:
        return None
    
    scenario_id = scenario_match.group(1)
    actor_part = prefix[7:]
    
    # Extract actor and number
    actor_match = re.match(r'^([A-Z]+)(\d+)', actor_part)
    if actor_match:
        actor = actor_match.group(1)
        instance_num = actor_match.group(2)
    else:
        actor = actor_part[0] if actor_part else "?"
        instance_num = actor_part[1:] if len(actor_part) > 1 else "?"
    
    return {
        "scenario_id": scenario_id,
        "actor": actor,
        "instance_num": instance_num,
        "modality": modality,  # V = Visual (face), T = Textual (voice)
        "emotion": emotion,
    }

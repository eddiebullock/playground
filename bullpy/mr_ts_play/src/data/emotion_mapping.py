"""
Map 410 fine-grained emotions to 6-7 basic emotion categories.
Uses Ekman's 6 basic emotions plus Neutral.
"""

# Ekman's 6 basic emotions + Neutral
BASIC_EMOTIONS = {
    'happy': 0,
    'sad': 1,
    'angry': 2,
    'fear': 3,
    'surprise': 4,
    'disgust': 5,
    'neutral': 6,
}

# Reverse mapping for display
EMOTION_NAMES = {v: k for k, v in BASIC_EMOTIONS.items()}


def create_emotion_mapping():
    """
    Create mapping from fine-grained emotions to basic categories.
    This is a heuristic mapping - you may want to refine it.
    """
    mapping = {}
    
    # Happy category
    happy_keywords = [
        'happy', 'joy', 'glad', 'pleased', 'delighted', 'cheerful', 'merry',
        'jubilant', 'elated', 'ecstatic', 'overjoyed', 'gleeful', 'cheered',
        'cheering', 'lively', 'playful', 'excited', 'enthusiastic', 'exhilarated',
        'grateful', 'content', 'satisfied', 'comfortable', 'relaxed', 'calm',
        'peaceful', 'serene', 'tranquil', 'safe', 'secure', 'confident',
        'proud', 'triumphant', 'victorious', 'successful', 'accomplished',
        'appreciated', 'valued', 'loved', 'adored', 'cherished', 'treasured',
        'welcome', 'welcoming', 'included', 'accepted', 'belonging',
        'hopeful', 'optimistic', 'positive', 'upbeat', 'bright',
        'lucky', 'fortunate', 'blessed', 'privileged',
    ]
    
    # Sad category
    sad_keywords = [
        'sad', 'sorrow', 'grief', 'mourning', 'melancholy', 'depressed',
        'down', 'low', 'blue', 'unhappy', 'miserable', 'wretched',
        'heartbroken', 'heartache', 'devastated', 'crushed', 'broken',
        'lonely', 'isolated', 'abandoned', 'deserted', 'neglected',
        'disappointed', 'let down', 'disillusioned', 'disheartened',
        'hopeless', 'despairing', 'desperate', 'helpless', 'powerless',
        'hurt', 'wounded', 'injured', 'damaged', 'bruised',
        'tearful', 'crying', 'weeping', 'sobbing', 'bawling',
        'gloomy', 'grim', 'dreary', 'bleak', 'dark',
        'tired', 'exhausted', 'weary', 'drained', 'spent',
        'homesick', 'nostalgic', 'longing', 'yearning', 'pining',
    ]
    
    # Angry category
    angry_keywords = [
        'angry', 'mad', 'furious', 'rage', 'wrath', 'ire',
        'annoyed', 'irritated', 'frustrated', 'aggravated', 'exasperated',
        'hostile', 'aggressive', 'violent', 'fierce', 'furious',
        'resentful', 'bitter', 'spiteful', 'vindictive', 'revengeful',
        'hateful', 'hatred', 'hated', 'detesting', 'despising',
        'contemptuous', 'scornful', 'disdainful', 'disrespectful',
        'defiant', 'rebellious', 'stubborn', 'obstinate',
        'provoked', 'incited', 'agitated', 'riled', 'worked up',
        'heated', 'passionate', 'intense', 'fiery', 'explosive',
        'judgmental', 'condemning', 'criticizing', 'blaming', 'accusing',
        'argumentative', 'combative', 'confrontational', 'challenging',
    ]
    
    # Fear category
    fear_keywords = [
        'fear', 'afraid', 'scared', 'frightened', 'terrified', 'horrified',
        'panic', 'panicked', 'alarmed', 'startled', 'shocked',
        'anxious', 'worried', 'nervous', 'uneasy', 'apprehensive',
        'dread', 'dreading', 'dreadful', 'fearful', 'timid',
        'intimidated', 'threatened', 'threatening', 'menacing',
        'terrorized', 'terror', 'horror', 'horrified',
        'vulnerable', 'exposed', 'defenseless', 'helpless',
        'cowardly', 'cowed', 'timid', 'shy', 'bashful',
        'hesitant', 'uncertain', 'unsure', 'doubtful', 'suspicious',
    ]
    
    # Surprise category
    surprise_keywords = [
        'surprise', 'surprised', 'shocked', 'startled', 'astonished',
        'amazed', 'astounded', 'stunned', 'bewildered', 'baffled',
        'confused', 'puzzled', 'perplexed', 'mystified', 'bewildered',
        'disbelieving', 'incredulous', 'skeptical', 'doubtful',
        'questioning', 'curious', 'wondering', 'inquiring',
        'alert', 'attentive', 'vigilant', 'watchful', 'observant',
    ]
    
    # Disgust category
    disgust_keywords = [
        'disgust', 'disgusted', 'revolted', 'repulsed', 'repelled',
        'sickened', 'nauseated', 'queasy', 'squeamish',
        'appalled', 'horrified', 'shocked', 'scandalized',
        'contempt', 'contemptuous', 'scorn', 'scornful',
        'revulsion', 'aversion', 'distaste', 'loathing',
        'sarcastic', 'cynical', 'skeptical', 'doubtful',
    ]
    
    # Neutral category (emotions that don't fit clearly into above)
    neutral_keywords = [
        'neutral', 'calm', 'composed', 'serene', 'peaceful',
        'indifferent', 'unconcerned', 'detached', 'distant', 'remote',
        'blank', 'empty', 'vacant', 'vague', 'unfocused',
        'thinking', 'thoughtful', 'contemplative', 'reflective',
        'listening', 'attentive', 'focused', 'concentrating',
        'decided', 'determined', 'resolved', 'committed',
        'sure', 'certain', 'confident', 'convinced',
        'patient', 'tolerant', 'accepting', 'understanding',
        'polite', 'respectful', 'courteous', 'civil',
        'honest', 'sincere', 'genuine', 'authentic',
        'modest', 'humble', 'unassuming', 'self-effacing',
    ]
    
    # Create mapping dictionary
    all_keywords = {
        'happy': happy_keywords,
        'sad': sad_keywords,
        'angry': angry_keywords,
        'fear': fear_keywords,
        'surprise': surprise_keywords,
        'disgust': disgust_keywords,
        'neutral': neutral_keywords,
    }
    
    # Build mapping: for each fine-grained emotion, find which basic category it belongs to
    # This will be populated by analyzing the dataset
    return all_keywords


def map_emotion_to_basic(fine_grained_emotion: str, keyword_mapping: dict = None) -> str:
    """
    Map a fine-grained emotion to a basic emotion category.
    
    Args:
        fine_grained_emotion: The fine-grained emotion name (e.g., "humiliated")
        keyword_mapping: Optional pre-computed mapping
    Returns:
        Basic emotion category (e.g., "sad")
    """
    if keyword_mapping is None:
        keyword_mapping = create_emotion_mapping()
    
    emotion_lower = fine_grained_emotion.lower()
    
    # Check each category for keyword matches
    for category, keywords in keyword_mapping.items():
        for keyword in keywords:
            if keyword in emotion_lower or emotion_lower in keyword:
                return category
    
    # Default to neutral if no match
    return 'neutral'


def create_emotion_mapping_from_dataset(dataset_path: str) -> dict:
    """
    Create emotion mapping by analyzing the dataset.
    This helps identify which emotions exist and how to map them.
    """
    import pandas as pd
    from pathlib import Path
    
    # Load all emotions from dataset
    emotions = set()
    data_root = Path(dataset_path)
    
    for emotion_folder in sorted(data_root.glob("[0-9][0-9]")):
        if not emotion_folder.is_dir():
            continue
        for scenario_folder in sorted(emotion_folder.glob("[0-9]*")):
            if not scenario_folder.is_dir():
                continue
            for video_file in scenario_folder.glob("*.mov"):
                # Parse emotion from filename
                import re
                base = video_file.name.replace(".mov", "")
                match = re.search(r'([VT])([a-z]+(?:-[a-z]+)*)$', base)
                if match:
                    emotion = match.group(2)
                    emotions.add(emotion)
    
    # Create mapping for each emotion
    keyword_mapping = create_emotion_mapping()
    emotion_to_basic = {}
    
    for emotion in sorted(emotions):
        basic = map_emotion_to_basic(emotion, keyword_mapping)
        emotion_to_basic[emotion] = basic
    
    return emotion_to_basic


if __name__ == "__main__":
    # Test the mapping
    test_emotions = [
        'humiliated', 'ashamed', 'embarrassed', 'happy', 'sad', 'angry',
        'fearful', 'surprised', 'disgusted', 'calm', 'thinking'
    ]
    
    keyword_mapping = create_emotion_mapping()
    print("Emotion Mapping Test:")
    print("-" * 60)
    for emotion in test_emotions:
        basic = map_emotion_to_basic(emotion, keyword_mapping)
        print(f"{emotion:20s} -> {basic}")






